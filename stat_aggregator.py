import os
import warnings
from pprint import pprint

import pandas as pd

warnings.simplefilter("ignore")


class StatAggregator:
    def __init__(self, directory, extra_file):
        self.directory = directory
        self.extra_file = extra_file
        self.store = {}

    def iterate_directory(self):
        # iterate over files in that directory
        for filename in os.listdir(self.directory):
            f = os.path.join(self.directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                if "extras" not in f and "xlsx" in f:
                    print(f)
                    self.get_stats(f)

    def get_extras(self):
        data = pd.read_excel(
            self.extra_file,
            engine="openpyxl",
            keep_default_na=False,
        )

        for name, record in data.iterrows():
            name = record["ΟΝΟΜΑ"]
            if name not in self.store:
                self.store[name] = {}
                self.store[name]["month_count"] = 0
                self.store[name]["months"] = {}
                self.store[name]["ΕΑ"] = 0  # καβαρντινας...
                self.store[name]["extras"] = 0  # randoms...
            self.store[name]["extras"] = record["ΧΡΕΩΜΕΝΕΣ"] + record["ΕΒΡΟΣ"] * 2

    def get_stats(self, file):
        data = pd.read_excel(
            file,
            engine="openpyxl",
            keep_default_na=False,
        )

        for name, record in data.iterrows():
            name = record["ΟΝΟΜΑ"]
            del record["ΟΝΟΜΑ"]
            if "sum" in record:
                del record["sum"]
            for date, task in record.items():
                print(date)
                parts = date.split("_")
                month = parts[0]
                day = parts[1]
                day_type = parts[2]
                if name not in self.store:
                    self.store[name] = {}
                    self.store[name]["month_count"] = 0
                    self.store[name]["months"] = {}
                    self.store[name]["ΕΑ"] = 0  # καβαρντινας...
                    self.store[name]["extras"] = 0  # randoms...
                    self.store[name]["total"] = 0  # randoms...
                if month not in self.store[name]["months"]:
                    self.store[name]["months"][month] = {}
                if day_type not in self.store[name]:
                    self.store[name][day_type] = 0

                if task not in ["Χ", "!", "?", "ΕΦ", "", "Ε", "Α"]:
                    self.store[name][day_type] += 1
                    self.store[name]["months"][month][day] = task
                    if task not in self.store[name]:
                        self.store[name][task] = 0
                    self.store[name][task] += 1

            # inc month count
            self.store[name]["month_count"] += 1

            # print(name)
            # pprint(self.store[name])

    def get_totals(self):
        dts = ["Κ", "ΠΑ", "ΠΑΤ", "ΕΑ", "ΕΑΤ", "Α", "ΑΤ"]
        for k, v in self.store.items():
            for x in dts:
                if x not in v:
                    v[x] = 0
            if self.store[k]["month_count"] != 0:
                self.store[k]["total"] = sum([v[x] for x in dts])

                self.store[k]["%"] = (
                    self.store[k]["total"] / self.store[k]["month_count"]
                )
                self.store[k]["Α%"] = (
                    self.store[k]["Α"] + self.store[k]["ΑΤ"]
                ) / self.store[k]["month_count"]
                self.store[k]["ΕΑ%"] = (
                    self.store[k]["ΕΑ"] + self.store[k]["ΕΑΤ"]
                ) / self.store[k]["month_count"]
                self.store[k]["Τ%"] = (
                    self.store[k]["ΕΑΤ"] + self.store[k]["ΑΤ"]
                ) / self.store[k]["month_count"]
                self.store[k]["adj%"] = (
                    self.store[k]["extras"] + self.store[k]["total"]
                ) / self.store[k]["month_count"]

    def print_totals(self):
        print(
            "ΟΝΟΜΑ, ΜΗΝΕΣ, ΣΥΝΟΛΟ, ΣΥΝΟΛΟ_ΑΡΓΙΩΝ, ΣΥΝΟΛΟ ΠΑΡΑΣΚΕΥΩΝ, ΕΠΙΣΗΜΕΣ_ΑΡΓΙΕΣ, ΥΠΗΡΕΣΙΕΣ_ΤΡΙΗΜΕΡΟΥ, ΕΒΡΟΣ"
        )
        for k, v in self.store.items():
            if "total" in v:
                print(
                    f"{k}, {v['month_count']}, {v['total']}, {v['Α']+v['ΑΤ']+v['ΕΑ']+v['ΕΑΤ']}, {v['ΠΑ']+v['ΠΑΤ']}, {v['ΕΑ']+v['ΕΑΤ']}, {v['ΕΑΤ']+v['ΑΤ']}"
                )

    def print_specific(self, duty):
        for k, v in self.store.items():
            sum = 0
            for dt, val in v.items():
                if duty in dt:
                    sum += val
            print(k, sum, round(sum / v["month_count"], 2))

    def get_store(self):
        self.iterate_directory()
        self.get_extras()
        self.get_totals()
        # self.print_totals()
        return self.store


def run():
    # directory = input("Give directory with files: ")
    aggregator = StatAggregator("previous_months", "extras.xlsx")
    aggregator.iterate_directory()
    aggregator.get_extras()
    aggregator.get_totals()
    aggregator.print_totals()
    # aggregator.print_specific("ΚΠ")


if __name__ == "__main__":
    run()
