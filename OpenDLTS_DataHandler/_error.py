class TransientDataTypeError(Exception):
    def __init__(self, message: str | None = None):
        if message is None:
            message = """
            class _TransientDataType_dict(TypedDict):
                t: np.typing.ArrayLike    # (Nt,)
                T: np.typing.ArrayLike    # (NT,)
                C: np.typing.ArrayLike    # (NT, Nt)
            TransientDataType = Path | str | None | _TransientDataType_dict | np.typing.ArrayLike  # (NT+1, Nt+1)
            """
        self.message = message
        super().__init__(self.message)
    def __str__(self):
        return self.message
