import statistics


def calculate_statistics(lst):
    maximum = max(lst)
    minimum = min(lst)
    average = sum(lst) / len(lst)
    standard_deviation = statistics.stdev(lst)
    return maximum, minimum, average, standard_deviation