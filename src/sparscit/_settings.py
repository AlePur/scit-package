class Config:
    """\
    Global settings for SparSCit.

    Attributes
    ----------
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    DEBUG : bool
        Whether to enable debug mode
    figsize : tuple[int, int]
        Default figure size for all plots
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
    Change global settings for SparSCit.

    Parameters
    ----------
    n_jobs
        Number of parallel jobs (-1 for all cores)
    DEBUG
        Whether to enable debug mode
    figsize
        Default figure size for all plots
    use_constrained_layout
        Whether to use constrained layout for matplotlib figures
    """
    if n_jobs is not None:
        settings.n_jobs = n_jobs
    if DEBUG is not None:
        settings.DEBUG = DEBUG
    if figsize is not None:
        settings.figsize = figsize
    if use_constrained_layout is not None:
        settings.constrained = use_constrained_layout