from collections import UserDict

class BidirectionalMap(UserDict):
    """
    A bidirectional map that allows for two-way lookups between keys and values.

    strictness = 0: Enforces a one-to-one mapping between keys and values. The user must ensure that no duplicate keys or values are added if not intended. In this version, keys or values no longer associated with a value or key will disappear.
    strictness = 1: Enforces a one-to-one mapping and raises an error if a duplicate key or value is added. In this version, you cannot overwrite an existing key with a new value or an existing value with a new key.
    """
    def __init__(self, *args, strictness=0, **kwargs):
        super().__init__()
        self._inverse = kwargs.pop('_inverse', None)
        if self._inverse is None:
            self._inverse = BidirectionalMap(_inverse=self, strictness=strictness)
        self.strictness = strictness
        if args or kwargs:
            self.update(*args, **kwargs)

    @property
    def inverse(self):
        return self._inverse
    
    @property
    def strictness(self):
        return self._strictness
    
    @strictness.setter
    def strictness(self, value):
        if value != 1 and value != 0:
            raise ValueError("Strictness must be either 0 or 1.")
        self._strictness = value
        self._inverse._strictness = value

    def __setitem__(self, key, value):
        if key in self:
            if self._strictness == 1 and self[key] != value:
                raise ValueError(f"Key '{key}' is already mapped to a different value '{self[key]}'.")
            del self._inverse.data[self[key]]
        if value in self._inverse:
            if self._strictness == 1 and self._inverse[value] != key:
                raise ValueError(f"Value '{value}' is already mapped to a different key '{self._inverse[value]}'.")
            del self.data[self._inverse[value]]

        self.data[key] = value
        self._inverse.data[value] = key

    def __delitem__(self, key):
        value = self[key]
        super().__delitem__(key)
        del self._inverse.data[value]

    def clear(self):
        super().clear()
        self._inverse.data.clear()

    def get_key(self, value):
        return self.inverse.get(value)

    def get_value(self, key):
        return self.get(key)


    
#############
### TESTS ###
#############
# TODO: put this in some separate file/subpackage.


def _check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


def _test_bidirectional_map():
    passed = 0
    total = 0

    def test(name, condition, detail=""):
        nonlocal passed, total
        total += 1
        if _check(name, condition, detail):
            passed += 1

    print("BidirectionalMap tests\n" + "=" * 30)

    # Construction
    bm = BidirectionalMap()
    test("starts empty", len(bm) == 0)
    test("inverse starts empty", len(bm.inverse) == 0)
    test("inverse inverse is original map", bm.inverse.inverse is bm)

    # Forward insertion
    bm["one"] = 1
    bm["two"] = 2
    bm["three"] = 3

    test("forward lookup", bm["one"] == 1)
    test("inverse lookup", bm.inverse[1] == "one")
    test("multiple forward values", dict(bm) == {"one": 1, "two": 2, "three": 3})
    test("multiple inverse values", dict(bm.inverse) == {1: "one", 2: "two", 3: "three"})

    # Inverse insertion
    bm.inverse[4] = "four"
    test("inverse assignment updates forward map", bm["four"] == 4)
    test("inverse assignment is readable", bm.inverse[4] == "four")

    # Replacing an existing key
    bm["one"] = 10
    test("key replacement updates forward value", bm["one"] == 10)
    test("key replacement adds new inverse value", bm.inverse[10] == "one")
    test("key replacement removes old inverse value", 1 not in bm.inverse)

    # Mapping API
    test("contains key", "two" in bm)
    test("contains inverse key", 2 in bm.inverse)
    test("get existing value", bm.get("two") == 2)
    test("get missing value", bm.get("missing") is None)
    test("get missing default", bm.get("missing", "default") == "default")
    test("keys view", set(bm.keys()) == {"one", "two", "three", "four"})
    test("values view", set(bm.values()) == {10, 2, 3, 4})
    test(
        "items view",
        set(bm.items()) == {("one", 10), ("two", 2), ("three", 3), ("four", 4)},
    )

    # Deletion from each direction
    del bm["two"]
    test("forward deletion removes forward item", "two" not in bm)
    test("forward deletion removes inverse item", 2 not in bm.inverse)

    del bm.inverse[3]
    test("inverse deletion removes inverse item", 3 not in bm.inverse)
    test("inverse deletion removes forward item", "three" not in bm)

    # Duplicate value behavior
    duplicate = BidirectionalMap(strictness=1)
    duplicate["first"] = 1

    try:
        duplicate["second"] = 1
    except Exception as error:
        print(f"[PASS] duplicate value is rejected — {type(error).__name__}: {error}")
        passed += 1
    else:
        print("[FAIL] duplicate value is rejected — two keys mapped to value 1")
    total += 1

    # Clear
    bm.clear()
    test("clear empties forward map", len(bm) == 0)
    test("clear empties inverse map", len(bm.inverse) == 0)

    print("=" * 30)
    print(f"Result: {passed}/{total} tests passed")