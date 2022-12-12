//+
Point(1) = {-0.075, 0, 0, 1.0};
//+
Point(2) = {-0.090, 0, 0, 1.0};
//+
Point(3) = {0, 0.075, 0, 1.0};
//+
Point(4) = {0, 0.090, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {3, 4};

//+
Point(5) = {0, 0, 0, 1.0};
//+
Circle(3) = {1, 5, 3};
//+
Circle(4) = {2, 5, 4};
//+
Curve Loop(1) = {2, -4, -1, 3};
//+
Plane Surface(1) = {1};
//+
Physical Curve("inner", 5) = {3};
//+
Physical Curve("outer", 6) = {4};
//+
Physical Curve("top", 7) = {2};
//+
Physical Curve("bot", 8) = {1};
//+
Physical Surface("surf", 9) = {1};
