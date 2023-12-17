class Feedback:
    def __init__(self):
        self._data = {}
        self._gdpr_ok = False

    def set_gdpr_ok(self):
        self._gdpr_ok = True

    def set_gdpr_not_ok(self):
        self._gdpr_ok = False

    def insert(self, key: str, value: str):
        self._data[key] = value

    def get_data(self):
        return self._data