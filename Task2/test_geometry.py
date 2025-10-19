import geometry

v1 = geometry.Vector2D(3.0, 4.0)
v2 = geometry.Vector2D(1.0, 2.0)

print(f"v1: {v1.toString()}")
print(f"Length of v1: {v1.length()}")   
v3 = v1.add(v2)
print(f"v1 + v2: {v3.toString()}")      
print(f"Dot product: {v1.dot(v2)}")     