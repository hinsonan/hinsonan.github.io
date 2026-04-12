---
layout: post
title: "Good Code vs Bad Code"
date: 2026-04-15
categories: ML
---

The question that has been debated by many overweight nerds and now new cool chad vibe coders is what is good code and bad code? You have the old guard zealots of uncle Bob's clean code that preach small functions and test driven development. You have the chads who don't look at the code and push based on vibes. One camp leaves you writing test suites that fall apart as soon as you code the real thing and the other leaks your credit card and crashes.

The truth of the matter is good code vs bad code is the wrong framework. The real issue is can you solve the problem in an efficient and elegant manner? Let's dive in

## Bad Code is Good and Good Code is Bad

One person's trash is another person's treasure and that's why bad code is simply code you yourself did not write. Perhaps it's code you wrote last month (or generated) and now your past self is here to haunt you. You are confronted with your sins and now you have to fix it. Many times what you think is bad code is really just a skill issue or lack of perception. For example you may look at the src code for Pytorch or React and think wow this sucks. There is some "bad" code in here. The people that work in that code and make decisions may think that code is good. To them it's good and to you it's bad. Other times it really is a poor solution but Pytorch is used by millions so it can't be complete garbage (However, having users doesn't mean it's good...looking at you Windows).

The other side of the coin is the zealots who insist clean code is the only way. Here is the problem with "clean code", it can make performance tank. You make all these nice little functions only for the end result to be slower. The other issue is you generally kill locality of scope and make me hop around to 50 different files. For those that are unaware here is some code and principles from the [book](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)

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

The "bad" code in this example stays hot in that sweet CPU cache and can be easily vectorized. The clean code forces pointer indirection and scatters all the objects across the heap which makes it harder to vectorize and lookup. The clean code example costs you performance. The counter argument to this is does this performance matter and isn't it better to have easy to maintain and easy to read code? To that I say what even is "easy to read code". As far as maintenance goes I prefer locality of scope and not making 100 tiny functions.

> PSA: At this point I sound a bit harsh towards clean code but it honestly is one of my favorite books and I like Uncle Bob. I think everyone should read it at different points in their career. Really adopt it and try hard to write "clean code". Then understand what you don't like about it. Read it as a junior, mid, senior developer and see how your thoughts have changed over time

Another fun example is the [inverse square root function](https://en.wikipedia.org/wiki/Fast_inverse_square_root) that allowed lighting in 3D graphics to work. Good luck explaining how this code works and it definitely does not fit "clean code"

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

The opposite is also true that highly optimized code is not always good code. If an extra data structure or variable could be declared and allow for developers to quickly debug a stack or heap issue at the tradeoff of a little extra server memory then it's probably worth it. 

So now that the waters are all muddy and we don't have a clear image of what is good code or bad code, how do we frame this topic. We all have worked in "bad code" and we see it a lot. We have wasted countless hours inside "bad code". Every so often you run across "good code" and can easily make the changes you need.

It's never been about good code vs bad code. It's about good and bad problem solving skills given the context and situation. That doesn't sound fancy and it doesn't make a good T-shirt. At the core it's always about problem solving.

## Code is the Vehicle for Problem Solving

Code is just the way that we express our solutions and problem solving. Truth be told the problem begins at communication and requirement gathering. I won't touch on this area of problem solving in this article but let's assume all the business meetings and customers have been established and it's time to solve and deliver the technical product.

Think of all the different areas of development:

* Research
* New Features
* Maintaining Existing Features
* Process Improvements
* CI/CD
* Development and Production Envs
* Infrastructure
* Proof of Concept Work
* Meetings
* System Design
* UX

There are many more areas and each area has its unique challenges. How can we establish what good problem solving is in a way that can encompass all the different duties we have to perform? It's simple: **Good problem solving is simply solving all the relevant problems given the constraints imposed on you at the start or during**

## Solving All the Worlds Problems

What leads to bad code is bad problem solving or just not solving the right types of problems at all. Let's talk about a new ML model or algorithm you are evaluating for a new feature that will be implemented soon.

You create this new algorithm in a Jupyter notebook and demonstrate that it does work as intended so you hand it off to the backend team to integrate. This leads to the backend team asking you questions and implementing that notebook into the system. Turns out the backend team took much longer to integrate into the system and you noticed that the next release doesn't even have your new algorithm running. They struggled to get it working with the system. You demonstrated the new technique works and from your perspective that was the end of the task. Truth is you solved only a handful of problems and you decided not to solve other more important problems.

We all know that getting something to work is just the first step of the problem. What people fail to grasp is when they clean it up and present their work they still failed to solve the real problems that will come later.

We mentioned constraints that are imposed on you when you start working on an issue and the one common chain that controls us is **time**. No one can beat Father Time. Time is always a major constraining factor. If you had unlimited time to create this new algorithm and all you had after years of work is one lousy broken script then you really suck. However if you are constrained to 2 days and it's a hard algorithm to create and you showed off a broken but sometimes working script then that might be very impressive given 2 days.

Here's a non exhaustive list of problems you need to solve to really make a good solution

* Does this solution work at the very basic level?
* Does this solution cover the major edge cases?
* Can others install or know how to use this new algorithm/feature
* Is there a developer guide or usage guide that walks people through how to integrate or use what you made?
* Did you optimize it for the performance requirements?
* Did you remove any barriers for the next group (Python model does not slot into a C#/Rust/Go backend)?
* Do you lay out the pros/cons of the way you solved the problem?
* Are the documents accurate to what the code does?
* Does this solve the business problems if this is relevant?
* How did you add or support debugging this algorithm/feature?
* Does this make future features and work easier?
* Does this pass CI/CD?

Most of the time you can't solve all of these problems in the amount of time you have. So you are forced to choose which of these problems is worth solving. So how do you know which of the problems to solve and which to reject? You have to communicate with your leads/team/managers to figure that out. You have to understand the problems they need to be solving so that you can know which problems and battles you need to fight.

This is why this field is so crazy. Once you see all these problems and all the constraints it gets hard to manage. This is why there is so much bad code out there. People are getting hounded to solve everything and things fall through the cracks. 

It gets more manageable when you have better communication with the team and know which battles are important to spend time on. Does your model's memory footprint need to be optimized if the team agrees that an 80% solution is perfectly acceptable for now? Probably not and it is better to provide how to use this new model and a basic inference guide. 

If you just hand them a notebook that semi-works, you did not solve the right problems for the team. If you spent days optimizing something but no one knows the API, you wasted time and failed.

## You ain't that Good

Here's another harsh truth: a lot of bad code is generated due to skill issues. Everything is a skill issue at the end of the day. You can't fix your car because you lack skills or you can't build a house due to lack of skills. You can't solve this technical problem all the way through due to constraints and skill. If you were a god and as good at programming as Terry Davis then you could make a novel algorithm in about one hour. Instead you could only solve the problem to the degree that your skills with the constraints allowed you.

When you see bad code and think oh I am so much better than whoever this person is you only think that because you did not solve that problem in the context that the other person had. Maybe that person had multiple meetings pop up or had to solve an emerging fire so they left the solution at 80% complete. Now that's not an excuse, it's just reality and maybe if you can improve it you should.

## Perspective is Everything

One thing that I have noticed in good problem solvers and good code is the person who wrote the code/system/docs put themselves in other people's shoes. If you can grasp a bird's-eye view of the project and understand how the manager thinks you can provide auto gen docs or system diagrams so they can show their boss. If you understand how the backend will work you can write your solution in a way that makes integration much faster. If you can reduce the system's complexity and make a design that requires less code changes over time then you made the whole team's future better.

It's all about being able to think and take on other people's perspective.

## Concrete Examples

### The API/Integration Mismatch

The person makes a working ML model with an inference example for the backend team to follow

#### Bad Code in this Context

```python
import pandas as pd
import numpy as np

def run_inference(raw_data):
    # hardcoded path — only works on one machine
    model = load_model("/home/dan/model.pkl")

    df = pd.DataFrame(raw_data)
    predictions = model.predict(df)

    results = pd.DataFrame({
        "user_id": df["user_id"],
        "score":   predictions,
        "rank":    np.argsort(predictions),
    })

    return results  # a DataFrame
```

This looks ok if the whole system is using Python and dataframes but if the backend is in a real language this will fail. `return results` is a dataframe that is in column order when the backend expects row ordering:

```json
{
  "user_id": {"0": 101, "1": 102, "2": 103},
  "score":   {"0": 0.91, "1": 0.45, "2": 0.78},
  "rank":    {"0": 2, "1": 0, "2": 1}
}
```

This format of columns will not be compatible in this scenario since the backend teams wanted rows. Even if this format could be easier to vectorize, it's not what the team agreed on.

**The Backend Side — attempt 1**

```go
type Prediction struct {
    UserID int64   `json:"user_id"`
    Score  float64 `json:"score"`
    Rank   int     `json:"rank"`
}

// expected a list of Prediction
var results []Prediction
json.Unmarshal(body, &results)

// results is nil — unmarshal silently failed.
// shape didn't match the struct.
processBatch(results[0]) // panic: index out of range
```

**Backend Side Attempt 2**

After looking at the Python code they write a workaround

```go
type RawDF struct {
    UserID map[string]int64   `json:"user_id"`
    Score  map[string]float64 `json:"score"`
    Rank   map[string]int     `json:"rank"`
}

// manually reassemble rows from the dict-of-dicts
for k, uid := range raw.UserID {
    row := Prediction{uid, raw.Score[k], raw.Rank[k]}
    // ...
}
```

This is brittle and breaks if a field changes or is added. No one documented the data contract so the backend team had to "reverse engineer" it.

Now it's time to fix the Python code

One issue is `np.int64` and `np.float32` are not JSON serializable. Depending on your serializer you either get a `TypeError` on the way out or a silent coercion into something unexpected. Explicitly casting with `int()` and `float()` can resolve this.

#### Better Code in this Context

```python
from dataclasses import dataclass, asdict
from typing import List
import os

@dataclass
class Prediction:
    user_id: int
    score:   float
    rank:    int

def run_inference(raw_data) -> List[Prediction]:
    model_path = os.environ["MODEL_PATH"]  # no hardcoded paths
    model = load_model(model_path)

    df = pd.DataFrame(raw_data)
    scores = model.predict(df)
    ranks  = scores.argsort()

    return [
        Prediction(
            user_id=int(row.user_id),   # cast numpy -> native Python
            score=float(scores[i]),     # json serializable
            rank=int(ranks[i])
        )
        for i, row in df.iterrows()
    ]
```

The JSON is now a list of records:

```json
[
  {"user_id": 101, "score": 0.91, "rank": 2},
  {"user_id": 102, "score": 0.45, "rank": 0},
  {"user_id": 103, "score": 0.78, "rank": 1}
]
```

The backend has almost no changes to make except for syntax:

```go
var results []Prediction
if err := json.Unmarshal(body, &results); err != nil {
    return err
}
processBatch(results)
```

This small example in Python is rewritten in a way that helps the backend team easily integrate this model. Now in a real system things would be much more complex and you would need to provide a user/dev guide on how to use and integrate.

### Extra Memory to Help Debugging

In this example we will see how "bad" code that makes more memory may be preferred in many cases.

#### Bad Code

This code has no intermediate variables and streams through memory one element at a time but if this code returns a wrong value or needs inspecting it will be difficult to determine if a user is eligible. Keep in mind this is not necessarily bad code it's just bad code if you expect this to need debugging and a critical path in the code.

```python
def run_pipeline(user_ids: list[int]) -> float:
    return sum(
        score * weight
        for score, weight in (
            (get_score(uid), get_weight(uid))
            for uid in user_ids
            if is_eligible(uid)
        )
    )
```

#### Better Code for Easier Debugging

This code has a larger memory footprint but if you need better logs and an easier debug process then this code will allow you to easily see the issues.

```python
from dataclasses import dataclass, field
from typing import Iterator
import logging

@dataclass
class PipelineStats:
    total:      int = 0
    eligible:   int = 0
    zero_weight: list = field(default_factory=list)
    min_score:  float = float("inf")
    max_score:  float = float("-inf")

def _score_stream(user_ids: list[int], stats: PipelineStats) -> Iterator[float]:
    for uid in user_ids:
        stats.total += 1
        if not is_eligible(uid):
            continue

        score  = get_score(uid)
        weight = get_weight(uid)
        stats.eligible += 1
        stats.min_score = min(stats.min_score, score)
        stats.max_score = max(stats.max_score, score)

        if weight == 0.0:
            stats.zero_weight.append(uid)  # only store outliers

        yield score * weight

def run_pipeline(user_ids: list[int], debug: bool = False) -> float:
    stats = PipelineStats()
    total = sum(_score_stream(user_ids, stats))

    logging.info(f"eligible: {stats.eligible}/{stats.total}, "
                 f"score range: [{stats.min_score:.2f}, {stats.max_score:.2f}]")

    if stats.zero_weight:
        logging.warning(f"zero-weight users (excluded from sum): {stats.zero_weight}")

    logging.debug(f"full stats: {stats}")

    return total
```

Yes this method is slightly more memory-heavy but the tradeoff is worth it if this breaks and you need to investigate the logs or step through the function. This is where your communication skills come in and you need to understand which tradeoffs are appropriate.

### Memory Layout and Padding

#### Bad Code

This is an example of a struct layout that will use more memory than needed.

```rust
#[repr(C)] // fixed field order for demonstration
struct PlayerState {
    is_alive:   bool,    // 1 byte
    score:      u64,     // 8 bytes, 8-byte alignment
    health:     u8,      // 1 byte
    level:      u32,     // 4 bytes, 4-byte alignment
    is_admin:   bool,    // 1 byte
    position_x: f64,    // 8 bytes, 8-byte alignment
    position_y: f64,    // 8 bytes
    flags:      u8,      // 1 byte
}
```

On a 64-bit machine, this layout is 56 bytes because the compiler inserts padding so aligned fields (`u64`, `f64`, `u32`) start at their proper offset. CPUs move data through cache lines (commonly 64 bytes), with individual loads/stores consisting of 1, 4, or 8 bytes.

We can reduce padding by reordering fields from largest alignment to smallest and by packing booleans into a flag when it is worth the complexity.

#### Better Code with Proper Alignment

```rust
#[repr(C)]
struct PlayerState {
    // 8-byte fields first
    score:      u64,     // bytes 0–7
    position_x: f64,    // bytes 8–15
    position_y: f64,    // bytes 16–23
    // 4-byte field next
    level:      u32,     // bytes 24–27
    // pack all 1-byte fields together
    health:     u8,      // byte  28
    flags:      u8,      // byte  29 (bitfield)
    // 2 bytes remain — no padding needed
    _pad:       [u8; 2], // bytes 30–31 (explicit)
}

// boolean flags packed into one u8
// instead of separate bool fields
const FLAG_ALIVE: u8 = 0b0000_0001;
const FLAG_ADMIN: u8 = 0b0000_0010;
const FLAG_IMMUNE:u8 = 0b0000_0100;

impl PlayerState {
    fn is_alive(&self)  -> bool { self.flags & FLAG_ALIVE != 0 }
    fn is_admin(&self)  -> bool { self.flags & FLAG_ADMIN != 0 }
    fn set_alive(&mut self) { self.flags |= FLAG_ALIVE; }
    fn set_dead(&mut self)  { self.flags &= !FLAG_ALIVE; }
}
```

On that same machine, this layout is 32 bytes instead of 56. Padding exists mostly to satisfy alignment and ABI rules. With this layout each store starts at it's properly aligned offset.

By moving the layout, aligned fields are placed at aligned offsets (8-byte fields at offsets divisible by 8).

This optimization requires some hardware and ABI awareness.

Remember that this is only appropriate to spend time on if this is a system that needs to be performant or a system that will be used by others. If this is throwaway code or part of something that will be thrown away then spending time on this is a waste.

## Conclusion

Good code and bad code are subjective and the correct way to think about it is good vs bad problem solving. Does your code solve all the relevant problems or does it ignore important problems.

Can you perceive your other team's needs and emotions? Do you know what barriers to bring down and what problems to focus on? If you can build the technical and communication skills to high levels then you will be writing better solutions that will lead to people wanting to work with you and saying you have good code.

If you fail to do this then you will continue to write bad code. Sometimes even if you are really good the constraints get to even the best and you will write bad code. Due to these crazy requirements that change on a whim bad code will continue to spew out from the keyboard.

Now it's easier than ever to write bad code because your favorite LLM can generate thousands of lines of code with bad solutions. Your LLM cannot solve or even comprehend some of the problems you have to tackle.

Strive to be a great problem solver and raise the bar for others around you.
