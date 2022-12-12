//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 1, 0, 1.0};
//+
Point(3) = {10., 1, 0, 1.0};
//+
Point(4) = {10., 0, 0, 1.0};
//+
Line(1) = {2, 1};
//+
Line(2) = {1, 4};
//+
Line(3) = {4, 3};
//+
Line(4) = {3, 2};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};

//+
Transfinite Surface {1} = {2, 1, 4, 3} Left;
//+
Transfinite Curve {4, 2} = 501 Using Progression 1;
//+
Transfinite Curve {1, 3} = 51 Using Progression 1;
