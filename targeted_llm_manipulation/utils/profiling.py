import cProfile
import random
from functools import wraps


def profile(output_file="profile_results.prof"):
    """
    Decorator to profile a function and save the results to a file. Use like this:

    @profile()
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            prof = cProfile.Profile()
            result = prof.runcall(func, *args, **kwargs)
            filename = str(random.random()) + output_file
            prof.dump_stats(filename)
            print("Profile saved to", filename)
            return result

        return wrapper

    return inner
