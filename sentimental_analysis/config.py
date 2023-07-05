import multiprocessing
import environ

@environ.config(prefix='')
class Config:
    """The common configuration for the project"""
    
    num_cpus = environ.var(default=multiprocessing.cpu_count())
    