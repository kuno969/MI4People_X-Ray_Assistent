from collections import defaultdict

class Feedback:
    def __init__(self) -> None:
        self._data = defaultdict(list)
        self._gdpr_ok = False

    def set_gdpr_ok(self) -> None:
        self._gdpr_ok = True

    def set_gdpr_not_ok(self) -> None:
        self._gdpr_ok = False

    def insert(self, key: str, value: str) -> None:
        self._data[key].append(value)

    def get_data(self) -> dict[str, str]:
        return self._data
