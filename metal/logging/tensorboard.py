from tensorboardX import SummaryWriter

from metal.logging.writer import LogWriter


class TensorBoardWriter(LogWriter):
    """Class for logging to Tensorboard during runs, as well as writing simple
    JSON logs at end of runs.

    Stores logs in log_dir/{YYYY}_{MM}_{DD}/{H}_{M}_{S}_run_name.json.
    """

    def __init__(
        self,
        out_dir=None,
        log_dir="tensorboard",
        writer_metrics=None,
        include_config=True,
    ):
        super().__init__(
            out_dir=out_dir,
            writer_metrics=writer_metrics,
            include_config=include_config,
        )

        # Set up TensorBoard summary writer
        self.tb_writer = SummaryWriter(self.log_subdir)

    def add_scalar(self, name, val, i):
        if super().add_scalar(name, val, i):
            if type(val) != list:
                self.tb_writer.add_scalar(name, val, i)

    def close(self):
        self.write()
        self.tb_writer.close()
