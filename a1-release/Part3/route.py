#!/usr/local/bin/python3
# route.py : Find routes through maps
#
# Code by: name IU ID
#
# Based on skeleton code by B551 Course Staff, Fall 2023
#


# !/usr/bin/env python3
import sys
import math
from queue import PriorityQueue

def citygps_read():
    city_gps_loc = {}
    with open('city-gps.txt') as file:
        for line in file:
            parts = line.strip().split(' ')
            city = parts[0]
            lat = float(parts[1])
            lon = float(parts[2])
            city_gps_loc[city] = (lat, lon)
    return city_gps_loc

def h_dist(ct1, ct2, city_gps_loc):
    lat1, lon1 = city_gps_loc[ct1]
    lat2, lon2 = city_gps_loc[ct2]

    radius = 3956
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)*2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)*2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = radius * c

    return distance

# A* algorithm for finding the shortest route based on distance
def astr_r_dist(start_city, end_city, city_gps_loc, d):
    p = PriorityQueue()
    v = set()
    ini_state = (start_city, 0, [])
    p.put(ini_state)
    
    while not p.empty():
        curnt_city, cost_so_far, path = p.get()

        if curnt_city == end_city:
            return path

        if curnt_city in v:
            continue

        v.add(curnt_city)

        for segment, (ngbhr, distance) in d.get(curnt_city, {}).items():
            
            new_cost = cost_so_far + distance
            
            h = h_dist(ngbhr, end_city, city_gps_loc)
            Tot_cost = new_cost + h
            p.put((ngbhr, new_cost, path + [(segment, distance)]))

    return []

# A* algorithm for finding the shortest route based on segments
def astr_routeSeg(start_city, end_city, d):
    p = PriorityQueue()
    v = set()

    ini_state = (start_city, 0, [])
    
    # Add the initial state to the p
    p.put(ini_state)
    
    while not p.empty():
        curnt_city, segments_so_far, path = p.get()
        
        # If the current city is the goal, return the path
        if curnt_city == end_city:
            return path
        # Skip this city if already v
        if curnt_city in v:
            continue
        # Mark the current city as v
        v.add(curnt_city)
        
        # Iterate through successor cities
        for segment, (ngbhr, distance, spdLimit) in d.get(curnt_city, {}).items():
            # Calculate the new number of segments to reach the ngbhr
            new_segmts = segments_so_far + 1
            
            # Add the ngbhr to the p with the updated segments and path
            p.put((ngbhr, new_segmts, path + [(segment, distance)]))
    
    
    return []


def get_route(start, end, cost):
    city_gps_loc = citygps_read()
    
    d = {}
    with open('road-segments.txt') as file:
        for line in file:
            parts = line.strip().split(' ')
            ct1 = parts[0]
            ct2 = parts[1]
            distance = float(parts[2])
            spdLimit = float(parts[3])
            segment = parts[4]
            
            if ct1 not in d:
                d[ct1] = {}
            if ct2 not in d:
                d[ct2] = {}
            
            d[ct1][segment] = (ct2, distance, spdLimit)
            d[ct2][segment] = (ct1, distance, spdLimit)
    
    if cost == 'distance':
        result = astr_r_dist(start, end, city_gps_loc, d)
    elif cost == 'segments':
        result = astr_routeSeg(start, end, d)
    elif cost == 'time':
        
        pass  
    elif cost == 'delivery':
        pass  
    else:
        raise Exception("Error: Invalid cost function")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise(Exception("Error: expected 3 arguments"))

    (_, start_city, end_city, cost_function) = sys.argv
    if cost_function not in ("segments", "distance", "time", "delivery"):
        raise(Exception("Error: invalid cost function"))

    result = get_route(start_city, end_city, cost_function)

    
    print("Start in %s" % start_city)
    
    for segment, distance in result:
        print("   Then go to %s via %s" % (segment, distance))

    print("\n          Total segments: %4d" % len(result))
    print("             Total miles: %8.3f" % sum(distance for _, distance in result))
    print("             Total hours: %8.3f" % (sum(distance for _, distance in result) / 65))
    print("Total hours for delivery: %8.3f" % (sum(distance for _, distance in result) / 65 + len(result) * 0.1))

