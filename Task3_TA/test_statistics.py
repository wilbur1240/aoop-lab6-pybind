import statistics

analyzer = statistics.DataAnalyzer()
analyzer.addValue(10.0)
analyzer.addValue(20.0)
analyzer.addValue(30.0)
analyzer.addValues([40.0, 50.0])

print(f"Count: {analyzer.getCount()}")
print(f"Mean: {analyzer.getMean()}")
print(f"Min: {analyzer.getMin()}")
print(f"Max: {analyzer.getMax()}")
print(f"Values: {analyzer.getValues()}")