# Copyright 2018 Harold Fellermann
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Stochastic simulation algorithms

This module provides implementations of different stochastic simulation
algorithms. All implementations have to implement the TrajectorySampler
interface by subclassing this abstract base class.
"""

import abc

from .utils import with_metaclass
from .structures import multiset
from .transitions import Event, Reaction


class TrajectorySampler(with_metaclass(abc.ABCMeta, object)):
    """Abstract base class for stochastic trajectory samplers.

    This is the interface for stochastic trajectory samplers, i.e.
    implementations of the stochastic simulation algorithm.
    A trajectory sampler is initialized with a given process and
    initial state, and optional start time, end time, and maximal
    number of iterations. The sampler instance can then be iterated
    over to produce a stochastic trajectory:

    >>> trajectory = DirectMethod(process, state, steps=10000)
    >>> for transition in trajectory:
    ...     print trajectory.time, trajectory.state, transition

    When implementing a novel TrajectorySampler, make sure to only
    generate random numbers using the TrajectorySampler.rng random
    number generator.
    """
    def __init__(self, process, state, t=0., tmax=float('inf'), steps=None, seed=None):
        """Initialize the sampler for the given Process and state.

        State is a dictionary that maps chemical species to positive
        integers denoting their copy number. An optional start time
        t, end time tmax, maximal number of steps, and random seed
        can be provided.
        """
        from random import Random

        if t < 0:
            raise ValueError("t must not be negative.")
        if tmax < 0:
            raise ValueError("tmax must not be negative.")
        if steps is not None and steps < 0:
            raise ValueError("steps must not be negative.")
        if any(n <= 0 for n in state.values()):
            raise ValueError("state copy numbers must be positive")

        self.process = process
        self.step = 0
        self.steps = steps
        self.time = t
        self.tmax = tmax
        self.rng = Random(seed)

        for transition in self.process.transitions:
            self.add_transition(transition)
        self.state = multiset()
        self.update_state(state)

    def update_state(self, dct):
        """Modify sampler state.

        Update system state and infer new applicable transitions.
        """
        for rule in self.process.rules:
            for trans in rule.infer_transitions(dct, self.state):
                trans.rule = rule
                self.add_transition(trans)
        self.state.update(dct)

    @abc.abstractmethod
    def add_transition(self, transition):
        """Add a new transition to the sampler

        Must be implemented by a subclass.
        """
        pass

    @abc.abstractmethod
    def prune_transitions(self):
        """Remove infered transitions that are no longer applicable.

        Must be implemented by a subclass.
        """
        pass

    @abc.abstractmethod
    def propose_potential_transition(self):
        """Propose new transition

        Must be implemented by a subclass.
        Must return a triple where the first item is the time of the
        proposed transition, the second item the transition itself,
        and the last item a tuple of additional arguments that are
        passed through to calls to TrajectorySampler.perform_transition.
        Time and transition have to be picked from the correct
        probability distributions.
        """
        return float('inf'), None, tuple()

    def is_applicable(self, time, transition, *args):
        """True if the transition is applicable
        
        The standard implementation always returns True, i.e. it assumes
        that any transition returned by propose_potential_transition is
        applicable. Overwrite this method if you want to implement
        accept/reject type of samplers such as composition-rejection.
        """
        return True

    def perform_transition(self, time, transition): # understand this part of the superclass
        """Perform the given transition.

        Sets sampler.time to time, increases the number of steps,
        performs the given transition by removing reactants and
        adding products to state, and calls Rule.infer_transitions
        for every rule of the process.
        If overwritten by a subclass, the signature can have additional
        arguments which are populated with the argument tuple returned
        by TrajectorySampler.propose_potential_transition.
        """
        self.step += 1
        self.time = time
        transition.last_occurrence = time
        self.state -= transition.true_reactants
        for rule in self.process.rules:
            for trans in rule.infer_transitions(transition.true_products, self.state):
                trans.rule = rule
                self.add_transition(trans)
        self.state += transition.true_products
        self.prune_transitions()

    def reject_transition(self, time, transition, *args):
        """Do not execute the given transition.
        
        The default implementation does nothing. Overwrite this method
        if you, for example, want to prevent the same transition from
        being proposed again.
        """
        pass

    def has_reached_end(self):
        """True if given max steps or tmax are reached."""
        return self.step == self.steps or self.time >= self.tmax

    def __iter__(self):
        """Standard interface to sample a stochastic trajectory.

        Yields each performed transition of the stochastic trajectory.

        This implementation first picks a potential transition.
        If the transition is applicable, it is performed, otherwise
        it is rejected. Iteration continues until a stop criterion
        occurs.
        Consider to overwrite self.propose_potential_transition,
        self.is_applicable, self.perform_transition,
        self.reject_transition or self.has_reached_end in favour of
        overloading __iter__ when implementing a TrajectorySampler.
        """
        while not self.has_reached_end():
            time, transition, args = self.propose_potential_transition()

            if time >= self.tmax:
                break
            elif not self.is_applicable(time, transition, *args): # time = "time reaction should occur", transition = "reaction that occurs",
                self.reject_transition(time, transition, *args)
            else:
                self.perform_transition(time, transition, *args)
                yield transition

        if self.step != self.steps and self.tmax < float('inf'):
            self.time = self.tmax


class DirectMethod(TrajectorySampler):
    """Implementation of Gillespie's direct method.

    The standard stochastic simulation algorithm, published in
    D. T. Gillespie, J. Comp. Phys. 22, 403-434 (1976).

    DirectMethod works only over processes that employ stochastic
    transitions whose propensity functions do not explicitly depend on
    time (autonomous Reaction's).

    DirectMethod maintains a list of current transitions and a list
    of propensities. When proposing transitions, propensities are
    calculated for all transitions and time and occuring transition
    are drawn from the appropriate probability distributions.

    See help(TrajectorySampler) for usage information.
    """
    def __init__(self, process, state, t=0., tmax=float('inf'), steps=None, seed=None): # initialise the data structures of the whole algoirthm
        if any(not isinstance(r, Reaction) for r in process.transitions):
            raise ValueError("DirectMethod only works with Reactions.")
        if any(not issubclass(r.Transition, Reaction) for r in process.rules):
            raise ValueError("DirectMethod only works with Reactions.")
        self.transitions = []
        self.propensities = []
        super(DirectMethod, self).__init__(process, state, t, tmax, steps, seed)

    def add_transition(self, transition): # not loop related
        self.transitions.append(transition)

    def prune_transitions(self): # takes away reaction with propensities = 0
        depleted = [
            i for i, (p, t) in enumerate(zip(self.propensities, self.transitions))
            if p == 0. and t.rule
        ]
        for i in reversed(depleted):
            del self.transitions[i]
            del self.propensities[i]

    def propose_potential_transition(self): # relates loop = pseudocode
        from math import log

        self.propensities = [r.propensity(self.state) for r in self.transitions] # relate to step 5
        total_propensity = sum(self.propensities) # relates to step 6
        if not total_propensity:
            return float('inf'), None, tuple()

        delta_t = -log(self.rng.random())/total_propensity #step 2 of the linear time algorithm

        transition = None
        pick = self.rng.random()*total_propensity
        for propensity, transition in zip(self.propensities, self.transitions):
            pick -= propensity # pick = pick - propensity
            if pick < 0.:
                break

        return self.time + delta_t, transition, tuple()


class CompositionRejection(TrajectorySampler):
    # Copyright 2018 Harold Fellermann
    #
    # Permission is hereby granted, free of charge, to any person obtaining
    # a copy of this software and associated documentation files
    # (the "Software"), to deal in the Software without restriction,
    # including without limitation the rights to use, copy, modify, merge,
    # publish, distribute, sublicense, and/or sell copies of the Software,
    # and to permit persons to whom the Software is furnished to do so,
    # subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be
    # included in all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    # EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    # MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    # IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    # CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    # TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    # SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """Stochastic simulation algorithms

    This module provides implementations of different stochastic simulation
    algorithms. All implementations have to implement the TrajectorySampler
    interface by subclassing this abstract base class.
    """

    import abc

    from .utils import with_metaclass
    from .structures import multiset
    from .transitions import Event, Reaction

    class TrajectorySampler(with_metaclass(abc.ABCMeta, object)):
        """Abstract base class for stochastic trajectory samplers.

        This is the interface for stochastic trajectory samplers, i.e.
        implementations of the stochastic simulation algorithm.
        A trajectory sampler is initialized with a given process and
        initial state, and optional start time, end time, and maximal
        number of iterations. The sampler instance can then be iterated
        over to produce a stochastic trajectory:

        >>> trajectory = DirectMethod(process, state, steps=10000)
        >>> for transition in trajectory:
        ...     print trajectory.time, trajectory.state, transition

        When implementing a novel TrajectorySampler, make sure to only
        generate random numbers using the TrajectorySampler.rng random
        number generator.
        """

        def __init__(self, process, state, t=0., tmax=float('inf'), steps=None, seed=None):
            """Initialize the sampler for the given Process and state.

            State is a dictionary that maps chemical species to positive
            integers denoting their copy number. An optional start time
            t, end time tmax, maximal number of steps, and random seed
            can be provided.
            """
            from random import Random

            if t < 0:
                raise ValueError("t must not be negative.")
            if tmax < 0:
                raise ValueError("tmax must not be negative.")
            if steps is not None and steps < 0:
                raise ValueError("steps must not be negative.")
            if any(n <= 0 for n in state.values()):
                raise ValueError("state copy numbers must be positive")

            self.process = process
            self.step = 0
            self.steps = steps
            self.time = t
            self.tmax = tmax
            self.rng = Random(seed)

            for transition in self.process.transitions:
                self.add_transition(transition)
            self.state = multiset()
            self.update_state(state)

        def update_state(self, dct):
            """Modify sampler state.

            Update system state and infer new applicable transitions.
            """
            for rule in self.process.rules:
                for trans in rule.infer_transitions(dct, self.state):
                    trans.rule = rule
                    self.add_transition(trans)
            self.state.update(dct)

        @abc.abstractmethod
        def add_transition(self, transition):
            """Add a new transition to the sampler

            Must be implemented by a subclass.
            """
            pass

        @abc.abstractmethod
        def prune_transitions(self):
            """Remove infered transitions that are no longer applicable.

            Must be implemented by a subclass.
            """
            pass

        @abc.abstractmethod
        def propose_potential_transition(self):
            """Propose new transition

            Must be implemented by a subclass.
            Must return a triple where the first item is the time of the
            proposed transition, the second item the transition itself,
            and the last item a tuple of additional arguments that are
            passed through to calls to TrajectorySampler.perform_transition.
            Time and transition have to be picked from the correct
            probability distributions.
            """
            return float('inf'), None, tuple()

        def is_applicable(self, time, transition, *args):
            """True if the transition is applicable

            The standard implementation always returns True, i.e. it assumes
            that any transition returned by propose_potential_transition is
            applicable. Overwrite this method if you want to implement
            accept/reject type of samplers such as composition-rejection.
            """
            return True

        def perform_transition(self, time, transition):  # understand this part of the superclass -> increases the number of steps and performs the transition by modifying the relevant reactancts and products
            """Perform the given transition.

            Sets sampler.time to time, increases the number of steps,
            performs the given transition by removing reactants and
            adding products to state, and calls Rule.infer_transitions
            for every rule of the process.
            If overwritten by a subclass, the signature can have additional
            arguments which are populated with the argument tuple returned
            by TrajectorySampler.propose_potential_transition.
            """
            self.step += 1 # self.step = self.step + 1
            self.time = time
            transition.last_occurrence = time
            self.state -= transition.true_reactants # self.state = self.state - transition.true_reactants
            for rule in self.process.rules:
                for trans in rule.infer_transitions(transition.true_products, self.state):
                    trans.rule = rule
                    self.add_transition(trans)
            self.state += transition.true_products # self.state = self.state + transition.true_products
            self.prune_transitions()

        def reject_transition(self, time, transition, *args):
            """Do not execute the given transition.

            The default implementation does nothing. Overwrite this method
            if you, for example, want to prevent the same transition from
            being proposed again.
            """
            pass

        def has_reached_end(self):
            """True if given max steps or tmax are reached."""
            return self.step == self.steps or self.time >= self.tmax

        def __iter__(self):
            """Standard interface to sample a stochastic trajectory.

            Yields each performed transition of the stochastic trajectory.

            This implementation first picks a potential transition.
            If the transition is applicable, it is performed, otherwise
            it is rejected. Iteration continues until a stop criterion
            occurs.
            Consider to overwrite self.propose_potential_transition,
            self.is_applicable, self.perform_transition,
            self.reject_transition or self.has_reached_end in favour of
            overloading __iter__ when implementing a TrajectorySampler.
            """
            while not self.has_reached_end():
                time, transition, args = self.propose_potential_transition()

                if time >= self.tmax:
                    break
                elif not self.is_applicable(time, transition,
                                            *args):  # time = "time reaction should occur", transition = "reaction that occurs",
                    self.reject_transition(time, transition, *args)
                else:
                    self.perform_transition(time, transition, *args)
                    yield transition

            if self.step != self.steps and self.tmax < float('inf'):
                self.time = self.tmax

    class DirectMethod(TrajectorySampler):
        """Implementation of Gillespie's direct method.

        The standard stochastic simulation algorithm, published in
        D. T. Gillespie, J. Comp. Phys. 22, 403-434 (1976).

        DirectMethod works only over processes that employ stochastic
        transitions whose propensity functions do not explicitly depend on
        time (autonomous Reaction's).

        DirectMethod maintains a list of current transitions and a list
        of propensities. When proposing transitions, propensities are
        calculated for all transitions and time and occuring transition
        are drawn from the appropriate probability distributions.

        See help(TrajectorySampler) for usage information.
        """

        def __init__(self, process, state, t=0., tmax=float('inf'), steps=None,
                     seed=None):  # initialise the data structures of the whole algoirthm
            if any(not isinstance(r, Reaction) for r in process.transitions):
                raise ValueError("DirectMethod only works with Reactions.")
            if any(not issubclass(r.Transition, Reaction) for r in process.rules):
                raise ValueError("DirectMethod only works with Reactions.")
            self.transitions = []
            self.propensities = []
            super(DirectMethod, self).__init__(process, state, t, tmax, steps, seed)

        def add_transition(self, transition):  # not loop related
            self.transitions.append(transition)

        def prune_transitions(self):  # takes away reaction with propensities = 0
            depleted = [
                i for i, (p, t) in enumerate(zip(self.propensities, self.transitions))
                if p == 0. and t.rule
            ]
            for i in reversed(depleted):
                del self.transitions[i]
                del self.propensities[i]

        def propose_potential_transition(self):  # relates loop = pseudocode
            from math import log

            self.propensities = [r.propensity(self.state) for r in self.transitions]  # relate to step 5
            total_propensity = sum(self.propensities)  # relates to step 6
            if not total_propensity:
                return float('inf'), None, tuple()

            delta_t = -log(self.rng.random()) / total_propensity  # step 2 of the linear time algorithm

            transition = None
            pick = self.rng.random() * total_propensity
            for propensity, transition in zip(self.propensities, self.transitions):
                pick -= propensity  # pick = pick - propensity
                if pick < 0.:
                    break

            return self.time + delta_t, transition, tuple()

    class CompositionRejection(TrajectorySampler):
        """Implementation of Gillespie's direct method."""

        def propose_potential_transition(self):  # relates loop = pseudocode
            from math import log

            self.propensities = [r.propensity(self.state) for r in self.transitions]  # relate to step 5
            total_propensity = sum(self.propensities)  # relates to step 6
            if not total_propensity:
                return float('inf'), None, tuple()

            delta_t = -log(self.rng.random()) / total_propensity  # step 2 of the linear time algorithm. In addition, self.rng.random() is a random number, denoted by r1.

            transition = None # this block of code relates to step 3a. """Group(G) is selected via binary search of the G values."""
            pmin = min(self.propensities)
            pmax = max(self.propensities)
            x = pmin
            count = 0
            boundaries = []
            while x < pmax:
                x = x * 2
                boundaries.append(x)
                count = count + 1
            Group = [] # Is the list for the group of propensities of reactions chosen by binary search
            r2ps = self.rng.random() * total_propensity
            for propensity, transition in zip(self.propensities, self.transitions):
                for i in boundaries:
                    if i/2 < r2ps < i:
                        for f in propensity:
                            if i/2 < f < i:
                                Group.append(f)

            r3 = self.rng.random() # this block of code relates to step 3b
            r4 = self.rng.random()
            a = len(self.propensities)
            m = max(self.propensities)
            b = 0:m
            if r3 in a:
                i = r3
            if r4 in b:
                r = r4
            for propensity, transition in Group: # searches in group selected by the composition approach of the prior block
                if propensity < r:
                    pass
                else: # if propensity is more or equal to r
                    break # terminates the current loop and resumes execution at the next statement.

            return self.time + delta_t, transition, tuple()
