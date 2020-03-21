import os
import logging
import functools

import cluster
import utils


class FileHandlerWithHeader(logging.FileHandler):
    """
    Logging FileHandler that adds a header to the first row of the log
    taken from https://stackoverflow.com/a/33492520/1002899
    """

    # Pass the file name and header string to the constructor.
    def __init__(self, filename, header, mode='a', encoding=None, delay=0):
        # Store the header information.
        self.header = header

        # Determine if the file pre-exists
        self.file_pre_exists = os.path.exists(filename)

        # Call the parent __init__
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)

        # Write the header if delay is False and a file stream was created.
        if not delay and self.stream is not None:
            self.stream.write('%s\n' % header)

    def emit(self, record):
        # Create the file stream if not already created.
        if self.stream is None:
            self.stream = self._open()

            # If the file pre_exists, it should already have a header.
            # Else write the header to the file so that it is the first line.
            if not self.file_pre_exists:
                self.stream.write('%s\n' % self.header)
                self.file_pre_exists = True

        # Call the parent class emit function.
        logging.FileHandler.emit(self, record)


class ClusterLogDecorator():
    """
    Log decorator for interactions.Simulation methods. Log format is csv
    Logs the state of the cluster based on self.level:
    * 1 - current time, and interactions.Simulation.t
    * 2 - add cluster kinetic, potetnital and total energy (expensive computation)
    * 3 - add position and velocity for up to 10 particles
    * 4 - add F for up to 10 particles, if decorating interactions.Simulation.calc_F


    :param name: name of the logger
    :type name: str
    :param fname: log filename
    :type fname: str
    :param fpath: pat
    :type fpath: str
    :param header: header placeholder, actual header is constructed based on level
    :type header: str
    :param before: if True, log before the function executes
    :type before: bool
    :param after: if True, log after the function executes
    :type after: bool
    :param level: log level
    :type level: int
    :param G: gravitational constant for calculating energy
    :type G: float
    """

    def __init__(self, name, fname, fpath, header, before, after, level, G):
        self.G = G
        self.level = level
        self.before = before
        self.after = after

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        fh = FileHandlerWithHeader(os.path.join(fpath, fname), header, delay=True)
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s:%(msecs)03d,%(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)

        logger.addHandler(fh)

        self.logger = logger

    def __call__(self, fn):
        @functools.wraps(fn)
        def decorated(*args, **kwargs):

            if not self.logger.handlers[0].file_pre_exists:
                # set up the header
                fields = ['time', 't']
                if self.level > 1:
                    fields += ['T', 'U', 'E']
                if self.level > 2:
                    # don't log more than 10 particles...
                    ids = utils.df_collectLimit(fn.__self__.cluster, 10, 'id', sortCol='id')
                    fields += [
                        f"x_{x['id']},y_{x['id']},z_{x['id']},vx_{x['id']},vy_{x['id']},vz_{x['id']},m_{x['id']}"
                        for x in ids]
                if self.level > 3:
                    fields += [
                        f"Fx_{x['id']},Fy_{x['id']},Fz_{x['id']}"
                        for x in ids]
                self.logger.handlers[0].header = ','.join(fields)

            def log(F_data=None):
                fields = [str(fn.__self__.t)]

                if self.level > 1:
                    T = cluster.calc_T(fn.__self__.cluster, self.G)
                    U = cluster.calc_U(fn.__self__.cluster, self.G)
                    fields += [str(T), str(U), str(T + U)]
                if self.level > 2:
                    data = utils.df_collectLimit(fn.__self__.cluster, 10,
                                                 "x", "y", "z", "vx", "vy", "vz", "m", sortCol='id')
                    fields += [str(item) for sublist in data for item in sublist]
                if self.level > 3:
                    if F_data:
                        fields += [str(item) for sublist in F_data for item in sublist]
                    else:
                        fields += [''] * len(data) * 3

                self.logger.info(','.join(fields))

            try:
                if self.before:
                    log()
                result = fn(*args, **kwargs)
                if self.after:
                    if fn.__name__ == 'calc_F' and self.level > 3:
                        F = utils.df_collectLimit(result, 10, 'Fx', 'Fy', 'Fz', sortCol='id')
                        log(F)
                    else:
                        log()
                return result
            except Exception as ex:
                self.logger.warning("Exception {0}".format(ex))
                raise ex
            return result

        return decorated
