import pickle

from targeted_llm_manipulation.environment_vectorized.trajectory_queue import EnvironmentGenerator


class PickleabilityChecker(pickle.Pickler):
    def persistent_id(self, obj):
        try:
            pickle.dumps(obj)
            return None
        except Exception as e:
            print(f"Cannot pickle object of type {type(obj)}: {e}")
            return obj


def check_pickleability(obj):
    try:
        with open("/dev/null", "wb") as f:
            p = PickleabilityChecker(f, protocol=pickle.HIGHEST_PROTOCOL)
            p.dump(obj)
        print("Object is picklable")
    except Exception as e:
        print(f"Object is not picklable: {e}")


if __name__ == "__main__":
    obj = EnvironmentGenerator("test")
    check_pickleability(obj)
