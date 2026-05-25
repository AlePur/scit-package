class Config:
    """\
    Global settings
    """
    n_jobs: int
    DEBUG: bool
    figsize: tuple[int, int]

    def __init__(self) -> None:
        self.n_jobs = -1
        self.DEBUG = False
        self.figsize = (10, 5)
        self.constrained = True

settings = Config()

def set_defaults(
        n_jobs: int | None = None,
        DEBUG: bool | None = None,
        figsize: tuple[int, int] | None = None,
        use_constrained_layout: bool | None = None
) -> None:
    """
    Change settings

    Parameters
    ----------
    n_jobs
    use_constrained_layout
    figsize
        Default figsize for all figures
    """
    if n_jobs is not None:
        settings.n_jobs = n_jobs
    if DEBUG is not None:
        settings.DEBUG = DEBUG
    if figsize is not None:
        settings.figsize = figsize
    if use_constrained_layout is not None:
        settings.constrained = use_constrained_layout