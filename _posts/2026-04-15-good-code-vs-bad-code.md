---
layout: post
title: "Good Code vs Bad Code"
date: 2026-04-15
categories: ML
---

The question that has been debated by many overweight nerds and now new cool chad vibe coders is what is good code and bad code? You have the old guard zealots of uncle Bob's clean code that preach small functions and test driven development. You have the chads who don't look at the code and push based on vibes. One camp leaves you writing test suites that fall apart as soon as you code to real thing and the other leaks your credit card and crashes.

The truth of the matter is good code vs bad code is the wrong argument. The real issue is can you solve the problem in an efficient and elegant manner? Let's dive in

## Bad Code is Good and Good Code is Bad

One persons trash is another persons treasure and that's why bad code is simply code you yourself did not write. Perhaps it's code your wrote last month (or generated) and now your past self is here to haunt you. You are confronted with your sins and now you have to fix it. Many times what you think is bad code is really just a skill issue or lack of perception. For example you may look at the src code for Pytorch or React and think wow this sucks. There is some "bad" code in here. The people that work in that code and make decisions may think that code is good. To them its good and to you its bad. Other times its really is a poor solution but Pytorch is used by millions so it can't be complete garbage (However users does not mean its good...looking at you Windows).

The other side of the coin is the zealots who insist clean code is the only way. Here is the problem with "clean code", it can make performance tank. You make all this nice little functions and test for the end result to be slower. The other issue is you generally kill locality of scope and make me hop around to 50 different files. For those that are unaware here is some code and principles from the [book](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)

**Principles**

1) Functions should do one thing — and be as small as possible (ideally 2–4 lines)

2) No boolean flag arguments — a flag means a function does more than one thing

3) Prefer polymorphism over if/switch statements

4) Follow the Law of Demeter — a class should know only its direct dependencies

5) DRY (Don't Repeat Yourself) — avoid duplication through abstraction

**Clean Code Style Example**

```c++
// Virtual dispatch — CPU can't predict
// which method to call. Pointer array
// kills cache locality.

class Shape {
  virtual double area() = 0;
};

class Circle : public Shape {
  double r;
  double area() { return 3.14 * r * r; }
};

class Rect : public Shape {
  double w, h;
  double area() { return w * h; }
};

// Array of pointers — heap scattered
Shape* shapes[N];
double total = 0;
for (auto* s : shapes)
  total += s->area(); // vtable lookup each time
```

**NOT Clean Code Style**

```c++
// All data packed flat in memory.
// One branch, CPU predicts it well.
// Compiler can auto-vectorize (SIMD).

enum ShapeKind { CIRCLE, RECT, TRI };

struct Shape {
  ShapeKind kind;
  double a, b; // reuse fields
};

Shape shapes[N]; // contiguous — cache hot
double total = 0;

for (auto& s : shapes) {
  switch (s.kind) {
    case CIRCLE: total += 3.14*s.a*s.a; break;
    case RECT:   total += s.a * s.b;     break;
    case TRI:    total += s.a * s.b/2;   break;
  }
}
```

The "bad" code in this example stays hot in that sweet cpu cache and can be easily vectorized. The clean code forces pointer indirection and scatters all the objects across the heap which makes it harder to vectorize and lookup. The clean code example cost you performance. The counter argument to this is does this performance matter and isn't it better to have easy to maintain and easy to read code? To that I say what even is "easy to read code". As far as maintenance goes I prefer locality of scope and not making 100 tiny functions.

> PSA: At this point I sound a bit harsh towards clean code but it honestly is one of my favorite books and I like Uncle Bob. I think everyone should read it at different points in their career. Really adopt it and try hard to right "clean code". Then understand what you don't like about it. Read it as a junior, mid, senior developer and see how your thoughts have changed over time

Another fun example is the [inverse square root function](https://en.wikipedia.org/wiki/Fast_inverse_square_root) that allowed lighting in 3D graphics to work. Good luck explaining how this code works and it defiantly does not fit "clean code"

This code includes the original comments made by the author.

```c
float Q_rsqrt( float number )
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;                       // evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
//	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y;
}
```

This code would be hard to maintain and explain. It's not easy to read but without this code you can kiss the birth of lighting engines in 3D graphics goodbye.

So now that the waters are all muddy and we don't have a clear image of what is good code or bad code how do we frame this topic. We all have worked in "bad code" and we see it a lot. We have wasted countless hours inside "bad code". Every so often you run across "good code" and can easily make the changes you need.

It's never been about good code vs bad code. It's about good and bad problem solving skills given the context and situation. That doesn't sound fancy and it doesn't make a good T-shirt. At the core it's always about problem solving.

# Code is the Vehicle for Problem Solving