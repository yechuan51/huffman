#!/bin/bash

# Initialize sum, min, and max variables
sum=0
min=1000000
max=0

# Run the program 100 times
for i in {1..100}
do
    # Execute the program and grep for the timing information
    time_ms=$(./archive ../15Mb.pdf | grep "Histograming" | awk '{print $3}')
    
    # Add the time to the sum
    sum=$(echo "$sum + $time_ms" | bc)
    
    # Check and update min and max if necessary
    if (( $(echo "$time_ms < $min" | bc -l) )); then
        min=$time_ms
    fi

    if (( $(echo "$time_ms > $max" | bc -l) )); then
        max=$time_ms
    fi
done

# Calculate the average
average=$(echo "$sum / 100" | bc -l)

echo "Average Histograming time: $average ms"
echo "Minimum Histograming time: $min ms"
echo "Maximum Histograming time: $max ms"
