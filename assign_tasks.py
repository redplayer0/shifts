import logging
import warnings
from pprint import pprint
from time import time

import pandas as pd
from ortools.linear_solver import pywraplp

from stat_aggregator import StatAggregator

warnings.simplefilter("ignore")

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y/%m/%d %I:%M:%S",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
    level=0,
)


def export_solution_to_xlsx(solution, pool, filename):
    pd.DataFrame(generate_list_of_lists(solution, pool)).to_excel(
        filename, header=False, index=False
    )


def generate_list_of_lists(solution, pool):
    logging.info("start generate_list_of_lists")
    list_of_lists = [["ΟΝΟΜΑ"] + [d for d in solution.keys()]]
    for p, person in pool.items():
        row = [p]
        for day, shifts in solution.items():
            d = int(day.split("_")[0])
            cell = ""
            if d in person["restrictions"]:
                cell = "Χ"
            elif d in person["choices"]:
                cell = "!"
            for shift in shifts:
                if p in shifts[shift]:
                    cell = shift
            row.append(cell)
        list_of_lists.append(row)

    logging.info("end generate_list_of_lists")
    return list_of_lists


def get_solution(x, pool, day_dict, shift_types_list):
    logging.info("start get_solution")
    solution = {}
    shifts = [s for s in shift_types_list.keys()]
    names = [n for n in pool.keys()]
    for (i, d, w), v in x.items():
        day = f"{d}_{''.join([x[0] for x in day_dict[d].split(' ')])}"
        if day not in solution:
            solution[day] = {}
        # Test if x[i,j] is 1 (with tolerance for floating point arithmetic)
        if v.solution_value() > 0.5:
            shift = shifts[w]
            name = names[i]
            if shift in solution[day]:
                solution[day][shift].append(name)
            else:
                solution[day][shift] = [name]
    logging.info("end get_solution")
    return solution


def generate_solution(solver, objectives):
    logging.info("start generate_solution")
    logging.info(f"Number of variables: {solver.NumVariables()}")
    logging.info(f"Number of constraints: {solver.NumConstraints()}")
    solver.Minimize(solver.Sum(objectives))
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        status = True
    else:
        status = False
    logging.info("end generate_solution")
    return status


def add_cost_objectives(x, solver, costs, objectives):
    logging.info("start add_cost_objectives")
    for index in x:
        objectives.append(costs[index] * x[index])
    logging.info("end add_cost_objectives")


def generate_initial_variables(x, solver, pool, day_dict, shift_types_list):
    logging.info("start generate_initial_variables")
    for i, (p, person) in enumerate(pool.items()):
        for w, (shift, sub_shifts) in enumerate(shift_types_list.items()):
            for values in sub_shifts:
                for j in range(
                    values["from"],
                    values["to"] + 1,
                ):
                    if shift in person["shift_types"]:
                        x[i, j, w] = solver.IntVar(0, 1, "")
    logging.info(f"Number of variables: {len(x)}")
    logging.info("end generate_initial_variables")


def generate_costs(pool, shift_types_list, day_dict):
    logging.info("start generate_costs")
    costs = {}
    for i, (p, person) in enumerate(pool.items()):
        for w, (shift, sub_shifts) in enumerate(shift_types_list.items()):
            for values in sub_shifts:
                for j in range(values["from"], values["to"] + 1):
                    if shift in person["shift_types"]:
                        day_type = day_dict[j]
                        dtypes = day_type.split()
                        costs[i, j, w] = 0
                        for dt in dtypes:
                            if dt in person:
                                costs[i, j, w] += person[dt]
    logging.info(f"Number of costs: {len(costs)}")
    logging.info("end generate_costs")
    # pprint(costs)
    # sleep(2)
    return costs


def generate_day_dict(
    first_day, last_day, pre_holidays, holidays, special_days, pre_threedays, threedays
):
    logging.info("start generate_day_dict")
    d = {}
    for day in range(first_day, last_day + 1):
        types = []
        if day in pre_holidays:
            types.append("preholiday")
        if day in holidays:
            types.append("holiday")
        if day in special_days:
            types.append("specialday")
        if day in pre_threedays:
            types.append("prethreeday")
        if day in threedays:
            types.append("threeday")
        if day not in set(
            pre_holidays + holidays + special_days + pre_threedays + threedays
        ):
            types.append("normal")
        d[day] = " ".join(types)
    logging.info("end generate_day_dict")
    return d


def process_shift_types_list(shift_types_list):
    logging.info("start process_shift_types_list")
    d = {}
    for shift in shift_types_list:
        if shift[0] not in d:
            d[shift[0]] = []

        d[shift[0]].append(
            {
                "count": int(shift[1]),
                "distance": int(shift[2]),
                "from": int(shift[3]),
                "to": int(shift[4]),
                "optional": shift[5],
            }
        )
    logging.info("end process_shift_types_list")
    # pprint(d)
    # sleep(1)
    return d


def generate_pool(
    shift_types_list, previously_assigned_shifts, names_with_data, day_dict
):
    logging.info("start populate_pool")
    pool = {}
    for name, record in names_with_data.iterrows():
        # check if person's shift types are a subset of all shift types
        # if not then do not add this person and continue to the next one
        if not set(record["ΥΠΗΡΕΣΙΑ"].split()) <= set(shift_types_list):
            continue
        pool[name] = {}
        pool[name]["shift_types"] = record["ΥΠΗΡΕΣΙΑ"].split()
        # will only set high if only one value is given else
        # if 2 values it will reverse the list from the split and assign them
        pool[name]["shift_count_limit"] = {
            k: int(lim)
            for k, lim in zip(["high", "low"], record["ΟΡΙΟ"].split("_")[::-1])
        }
        # set a low shift count limit in case one is not given
        if "low" not in pool[name]["shift_count_limit"]:
            pool[name]["shift_count_limit"]["low"] = 1
        pool[name]["holiday_count_limit"] = int(record["ΟΡΙΟ_ΑΡΓ"])
        pool[name]["restrictions"] = []
        pool[name]["choices"] = []
        pool[name]["maybe"] = []
        # here last day needs +1
        # because it refers to a cell name in excel
        for day in day_dict:
            match names_with_data.loc[name][str(day)]:
                case "Α":
                    pool[name]["restrictions"].append(day)
                case "Χ":
                    pool[name]["restrictions"].append(day)
                case "ΑΥΔΜ":
                    pool[name]["restrictions"].append(day)
                    pool[name]["restrictions"].append(day - 1)
                    pool[name]["restrictions"].append(day + 1)
                case "!":
                    pool[name]["choices"].append(day)
                case "?":
                    pool[name]["maybe"].append(day)

        pr = previously_assigned_shifts
        pr_names = previously_assigned_shifts.keys()
        # check if current name has been assigned before
        if name in pr_names:
            m = pr[name]["month_count"]
            if m != 0:
                pool[name]["normal"] = (pr[name]["Κ"] + pr[name]["extras"]) / m
                pool[name]["preholiday"] = pr[name]["ΠΑ"] / m
                pool[name]["prethreeday"] = pr[name]["ΠΑΤ"] / m
                pool[name]["holiday"] = pr[name]["Α"] / m
                pool[name]["threeday"] = (pr[name]["ΑΤ"] + pr[name]["ΕΑΤ"]) / m
                pool[name]["specialday"] = (pr[name]["ΕΑ"] + pr[name]["ΕΑΤ"]) / m
            else:
                pool[name]["normal"] = 0
                pool[name]["preholiday"] = 0
                pool[name]["prethreeday"] = 0
                pool[name]["holiday"] = 0
                pool[name]["threeday"] = 0
                pool[name]["specialday"] = 0
            # print(name, round(pr[name]["%"], 2))
    # input("END")
    logging.info("end generate_pool")
    for p, vals in pool.items():
        print(f"{p}: {vals['restrictions']},")
    return pool


def run(
    month=None,
    first_day=None,
    last_day=None,
    pre_holidays=[],
    holidays=[],
    special_days=[],
    pre_threedays=[],
    threedays=[],
    day_distance_between_shifts=1,
    shift_types_list={},
    # files
    file_with_restrictions=None,
    dir_with_previous_shift_assignements=None,
    file_with_extra_shifts=None,
    # rules
    daily_shift_type_count_rule=True,
    shift_count_limit_per_person_rule=True,
    one_shift_per_day_per_person_rule=True,
    personal_restrictions_rule=True,
    personal_choices_rule=True,
    at_most_one_special_day_per_person_rule=True,
    at_most_one_threeday_per_person_rule=True,
    at_most_one_preholiday_or_prethreeday_per_person_rule=True,
    personal_holiday_count_limit_rule=True,
    allow_optional_shifts_rule=True,
    distance_rule=True,
):
    logging.info("Initializing..")
    # generate a day dict with its day number as key and day type as value
    day_dict = generate_day_dict(
        first_day,
        last_day,
        pre_holidays,
        holidays,
        special_days,
        pre_threedays,
        threedays,
    )

    # process the shift_types_list
    shift_types_list = process_shift_types_list(shift_types_list)
    # get data about previously assigned shifts
    previously_assigned_shifts = StatAggregator(
        dir_with_previous_shift_assignements, file_with_extra_shifts
    ).get_store()

    # get planning phase data
    logging.info("start getting names_with_data from the file_with_restrictions")
    names_with_data = pd.read_excel(
        file_with_restrictions,
        engine="openpyxl",
        index_col=0,
        # skiprows=1,
        keep_default_na=False,
    )
    logging.info("end getting names_with_data from the file_with_restrictions")

    # populates the pool with people whose ["shift_types_list"]
    # is a subset of the shift_types_list
    pool = generate_pool(
        shift_types_list, previously_assigned_shifts, names_with_data, day_dict
    )

    # generate the costs for the model
    costs = generate_costs(pool, shift_types_list, day_dict)

    # instanciate a solver
    solver = pywraplp.Solver.CreateSolver("SCIP")
    # here we will store all variables
    x = {}
    # here we will store all objective terms
    objectives = []

    # generate initial variables based on the people
    # days we need the shift_type
    # and the shift_type itself
    generate_initial_variables(x, solver, pool, day_dict, shift_types_list)

    # check every rule and apply it
    if daily_shift_type_count_rule:
        apply_daily_shift_type_count_rule(x, solver, pool, day_dict, shift_types_list)
    if shift_count_limit_per_person_rule:
        apply_shift_count_limit_per_person_rule(
            x, solver, pool, day_dict, shift_types_list
        )
    if one_shift_per_day_per_person_rule:
        apply_one_shift_per_day_per_person_rule(
            x, solver, pool, day_dict, shift_types_list
        )
    if personal_restrictions_rule:
        apply_personal_restrictions_rule(x, solver, pool, shift_types_list)
    if personal_choices_rule:
        apply_personal_choices_rule(x, solver, pool, shift_types_list)
    if at_most_one_special_day_per_person_rule:
        apply_at_most_one_special_day_per_person_rule(
            x, solver, pool, special_days, shift_types_list
        )
    if at_most_one_threeday_per_person_rule:
        apply_at_most_one_threeday_per_person_rule(
            x, solver, pool, threedays, shift_types_list
        )
    if at_most_one_preholiday_or_prethreeday_per_person_rule:
        apply_at_most_one_preholiday_or_prethreeday_per_person_rule(
            x, solver, pool, pre_holidays, pre_threedays, shift_types_list
        )
    if personal_holiday_count_limit_rule:
        apply_personal_holiday_count_limit_rule(
            x, solver, pool, holidays, special_days, threedays, shift_types_list
        )
    if allow_optional_shifts_rule:
        apply_allow_optional_shifts_rule()

    if distance_rule:
        apply_distance_rule(
            x, solver, day_distance_between_shifts, pool, day_dict, shift_types_list
        )
    # add the basic cost objectives
    add_cost_objectives(x, solver, costs, objectives)

    status = generate_solution(solver, objectives)
    solution = get_solution(x, pool, day_dict, shift_types_list)

    export_solution_to_xlsx(solution, pool, "example.xlsx")

    logging.info("Exiting..")
    print()
    print("People:", len(pool))
    # print("Total variables:", len(x))
    # input("press enter to see solution..")
    print()
    pprint(solution)
    if status:
        print("Success")
    else:
        print("Failure")


def apply_daily_shift_type_count_rule_old(x, solver, pool, day_dict, shift_types_list):
    logging.info("start apply_daily_shift_type_count_rule")
    for w, (shift, values) in enumerate(shift_types_list.items()):
        for j in range(values["from"], values["to"] + 1):
            if values["optional"]:
                solver.Add(
                    solver.Sum(
                        [
                            x[i, j, w]
                            for i, (p, person) in enumerate(pool.items())
                            if shift in person["shift_types"]
                        ]
                    )
                    <= values["count"]
                )
            else:
                solver.Add(
                    solver.Sum(
                        [
                            x[i, j, w]
                            for i, (p, person) in enumerate(pool.items())
                            if shift in person["shift_types"]
                        ]
                    )
                    == values["count"]
                )
    logging.info("end apply_daily_shift_type_count_rule")


def apply_daily_shift_type_count_rule(x, solver, pool, day_dict, shift_types_list):
    logging.info("start apply_daily_shift_type_count_rule")
    for j in day_dict:
        for w, (s, sub_shifts) in enumerate(shift_types_list.items()):
            for shift in sub_shifts:
                if j in range(shift["from"], shift["to"] + 1):
                    solver.Add(
                        solver.Sum(
                            [
                                x[i, j, w]
                                for i, (p, person) in enumerate(pool.items())
                                if s in person["shift_types"]
                            ]
                        )
                        == shift["count"]
                    )
    logging.info("end apply_daily_shift_type_count_rule")


def apply_shift_count_limit_per_person_rule(
    x, solver, pool, day_dict, shift_types_list
):
    logging.info("start apply_shift_count_limit_per_person_rule")
    for i, (p, person) in enumerate(pool.items()):
        # set the high limit
        solver.Add(
            solver.Sum(
                [
                    x[i, j, w]
                    for w, (s, sub_shifts) in enumerate(shift_types_list.items())
                    for shift in sub_shifts
                    for j in range(shift["from"], shift["to"] + 1)
                    if s in person["shift_types"]
                ]
            )
            <= person["shift_count_limit"]["high"]
        )
        # set the low limit
        # to get an "exactly that" limit
        # set both low and high to the same value
        solver.Add(
            solver.Sum(
                [
                    x[i, j, w]
                    for w, (s, sub_shifts) in enumerate(shift_types_list.items())
                    for shift in sub_shifts
                    for j in range(shift["from"], shift["to"] + 1)
                    if s in person["shift_types"]
                ]
            )
            >= person["shift_count_limit"]["low"]
        )
    logging.info("end apply_shift_count_limit_per_person_rule")


def apply_one_shift_per_day_per_person_rule(
    x, solver, pool, day_dict, shift_types_list
):
    logging.info("start apply_one_shift_per_day_per_person_rule")
    for i, (p, person) in enumerate(pool.items()):
        for j in day_dict:
            try:
                partial = [
                    x[i, j, w]
                    for w, (s, sub_shifts) in enumerate(shift_types_list.items())
                    for shift in sub_shifts
                    if s in person["shift_types"]
                ] + [
                    x[i, j + 1, w]
                    for w, (s, sub_shifts) in enumerate(shift_types_list.items())
                    for shift in sub_shifts
                    if s in person["shift_types"]
                ]

                solver.Add(solver.Sum(partial) <= 1)
            except:
                pass
    logging.info("end apply_one_shift_per_day_per_person_rule")


def apply_personal_restrictions_rule(x, solver, pool, shift_types_list):
    logging.info("start apply_personal_restrictions_rule")
    for i, (p, person) in enumerate(pool.items()):
        for j in person["restrictions"]:
            solver.Add(
                solver.Sum(
                    [
                        x[i, j, w]
                        for w, (s, sub_shifts) in enumerate(shift_types_list.items())
                        for shift in sub_shifts
                        if j in range(shift["from"], shift["to"] + 1)
                        if s in person["shift_types"]
                    ]
                )
                == 0
            )
    logging.info("end apply_personal_restrictions_rule")


def apply_personal_choices_rule(x, solver, pool, shift_types_list):
    logging.info("start apply_personal_choices_rule")
    for i, (p, person) in enumerate(pool.items()):
        for j in person["choices"]:
            solver.Add(
                solver.Sum(
                    [
                        x[i, j, w]
                        for w, shift in enumerate(shift_types_list)
                        if shift in person["shift_types"]
                    ]
                )
                == 1
            )
    logging.info("end apply_personal_choices_rule")


def apply_at_most_one_special_day_per_person_rule(
    x, solver, pool, special_days, shift_types_list
):
    logging.info("start apply_at_most_one_special_day_per_person_rule")
    pprint(shift_types_list)
    for i, (p, person) in enumerate(pool.items()):
        solver.Add(
            solver.Sum(
                [
                    x[i, j, w]
                    for j in special_days
                    for w, (s, sub_shifts) in enumerate(shift_types_list.items())
                    for shift in sub_shifts
                    if j in range(shift["from"], shift["to"] + 1)
                    if s in person["shift_types"]
                ]
            )
            <= 1
        )
    logging.info("end apply_at_most_one_special_day_per_person_rule")


def apply_at_most_one_threeday_per_person_rule(
    x, solver, pool, threedays, shift_types_list
):
    logging.info("start apply_at_most_one_threeday_per_person_rule")
    for i, (p, person) in enumerate(pool.items()):
        solver.Add(
            solver.Sum(
                [
                    x[i, j, w]
                    for j in threedays
                    for w, (s, sub_shifts) in enumerate(shift_types_list.items())
                    for shift in sub_shifts
                    if j in range(shift["from"], shift["to"] + 1)
                    if s in person["shift_types"]
                ]
            )
            <= 1
        )
    logging.info("end apply_at_most_one_threeday_per_person_rule")


def apply_at_most_one_preholiday_or_prethreeday_per_person_rule(
    x, solver, pool, pre_holidays, pre_threedays, shift_types_list
):
    logging.info("start apply_at_most_one_preholiday_or_prethreeday_per_person_rule")
    for i, (p, person) in enumerate(pool.items()):
        solver.Add(
            solver.Sum(
                [
                    x[i, j, w]
                    for j in set(pre_holidays + pre_threedays)
                    for w, (s, sub_shifts) in enumerate(shift_types_list.items())
                    for shift in sub_shifts
                    if j in range(shift["from"], shift["to"] + 1)
                    if s in person["shift_types"]
                ]
            )
            <= 1
        )
    logging.info("end apply_at_most_one_preholiday_or_prethreeday_per_person_rule")


def apply_personal_holiday_count_limit_rule(
    x, solver, pool, holidays, special_days, threedays, shift_types_list
):
    logging.info("start apply_personal_holiday_count_limit_rule")
    for i, (p, person) in enumerate(pool.items()):
        # if a person is allowed to do more than 1 holiday
        # make sure he does only holidays and not threedays or special days
        if person["holiday_count_limit"] > 1:
            if set(special_days + threedays):
                solver.Add(
                    solver.Sum(
                        [
                            x[i, j, w]
                            for j in set(special_days + threedays)
                            for w, (s, sub_shifts) in enumerate(
                                shift_types_list.items()
                            )
                            for shift in sub_shifts
                            if j in range(shift["from"], shift["to"] + 1)
                            if s in person["shift_types"]
                        ]
                    )
                    == 0
                )

        partial = [
            x[i, j, w]
            for j in set(holidays)
            for w, (s, sub_shifts) in enumerate(shift_types_list.items())
            for shift in sub_shifts
            if j in range(shift["from"], shift["to"] + 1)
            if s in person["shift_types"]
        ]
        solver.Add(solver.Sum(partial) <= person["holiday_count_limit"])
    logging.info("end apply_personal_holiday_count_limit_rule")


def apply_distance_rule(
    x, solver, day_distance_between_shifts, pool, day_dict, shift_types_list
):
    logging.info("start apply_distance_rule")
    for i, (p, person) in enumerate(pool.items()):
        for w, (s, shift) in enumerate(shift_types_list.items()):
            for w, (s, sub_shifts) in enumerate(shift_types_list.items()):
                for shift in sub_shifts:
                    if s in person["shift_types"]:
                        for j in range(
                            shift["from"], shift["to"] + 1 - shift["distance"]
                        ):
                            sum = []
                            for d in range(shift["distance"] + 1):
                                # if d is in person's choices do not consider it
                                # for the application of the distance rule
                                if j + d not in person["choices"]:
                                    sum.append(x[i, j + d, w])
                            solver.Add(solver.Sum(sum) <= 1)

    logging.info("end apply_distance_rule")


def apply_allow_optional_shifts_rule():
    pass


def test():
    # TODO run separately for every shift in the list
    # TODO run with distance from 6 to 1 until we have success
    run(
        month=11,
        first_day=1,
        last_day=30,
        pre_holidays=[3, 10, 17, 20, 24],
        holidays=[4, 5, 11, 12, 18, 19, 25, 26],
        special_days=[21],
        pre_threedays=[],
        threedays=[],
        day_distance_between_shifts=1,
        shift_types_list=[
            ("ΕΑΑΣ", 1, 2, 1, 30, False),
            # ("ΑΥΔΜ", 1, 1, 2, 5, False),
            # ("ΑΥΔΜ", 1, 1, 7, 7, False),
            # ("ΑΥΔΜ", 1, 1, 10, 11, False),
            # ("ΑΥΔΜ", 1, 1, 14, 14, False),
            # ("ΑΥΔΜ", 1, 1, 16, 22, False),
            # ("ΑΥΔΜ", 1, 1, 24, 29, False),
            # ("ΑΥΔΜ", 1, 1, 31, 31, False),
            # ("ΒΑΥΔΜ", 1, 2, 1, 31, False),
            # ("ΚΠ1", 1, 1, 1, 31, False),
            # ("ΚΠ2", 1, 1, 1, 30, False),
            # ("ΣΚ", 3, 1, 1, 12, False),
            # ("ΣΚ", 2, 1, 13, 14, False),
            # ("ΣΚ", 2, 1, 16, 16, False),
            # ("ΣΚ", 1, 1, 17, 31, False),
            # ("ΦΥΛ", 1, 1, 21, 23, False),
        ],
        # files
        file_with_restrictions="nov_23.xlsx",
        dir_with_previous_shift_assignements="previous_months",
        file_with_extra_shifts="extras.xlsx",
        # rules
        daily_shift_type_count_rule=True,
        shift_count_limit_per_person_rule=True,
        one_shift_per_day_per_person_rule=True,
        personal_restrictions_rule=True,
        personal_choices_rule=True,
        at_most_one_special_day_per_person_rule=True,
        at_most_one_threeday_per_person_rule=True,
        at_most_one_preholiday_or_prethreeday_per_person_rule=True,
        personal_holiday_count_limit_rule=True,
        allow_optional_shifts_rule=True,
        distance_rule=True,
    )


start = time()
test()
total = time() - start
print("Total time:", total)
