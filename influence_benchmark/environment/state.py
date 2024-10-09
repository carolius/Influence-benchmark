import copy


class State:
    def __init__(
        self,
        name: str,
        history: list,
        format_vars: dict,
        turns: int,
        valid_transitions: dict,
        default_transition: str,
        terminal: bool = False,
    ):
        """
        Initialize a State instance.

        Args:
            name (str): The name of the state.
            history (list): The conversation history.
            format_vars (dict): Variables used for formatting messages.
            turns (int): The number of turns taken.
            valid_transitions (dict): Valid state transitions.
            default_transition (str): The default transition if no valid transition is found.
            terminal (bool, optional): Whether this is a terminal state. Defaults to False.
        """
        self.name = name
        self.history = history
        self.format_vars = format_vars
        self.turns = turns
        self.valid_transitions = valid_transitions
        self.default_transition = default_transition
        self.terminal = terminal
        self.preferences = {}
        self.influence_scores = {}
        self.transition_probs = {}

    def copy(self):
        return State(  # important to use deepcopy as history is a list of dictionaries which are mutable
            self.name,
            copy.deepcopy(self.history),
            self.format_vars,
            self.turns,
            copy.deepcopy(self.valid_transitions),
            self.default_transition,
            self.terminal,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def __str__(self) -> str:
        return f"History: {self.history}, Variables: {self.format_vars}, Turns: {self.turns}"

    def __getitem__(self, key):
        return getattr(self, key)
