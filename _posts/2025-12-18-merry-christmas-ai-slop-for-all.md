---
layout: post
title: "Merry Christmas: Santa is Bringing AI Slop"
date: 2025-12-18
categories: ML
---

Ho Ho Ho to all vibe coders and all those failing production pipelines. Santa is coming to town and this year he's checking his AI generated naughty and nice list and didn't review it before hand. His fat self is sliding down your dirty chimney and your present this year is a toy that looks correct at a glance but the more you look at it you realize Santa just gave you Mrs. Clause's faulty elf action figure with 12 fingers and a message attached that says "LGTM I pushed this +500 changes through to prod. My agent workflow told me it was high quality". Santa climbs his way to the roof and heads off but instead of doing his normal route that uses `A*` search and plans an optimal path he is using a new sleek agent workflow that charted his path. The agents said he was absolutely right about this being better than old crusty `A*`. The only issue is it cost thousands of dollars to make this new route and the agent workflow had a hiccup and kept saying Santa isn't real and refused to make a route for him to follow. Now due to this new expensive route planning tool he had to layoff some elves and he is now vibing his way through your neighborhoods with no plan. Your fake girlfriend doll is never coming but it's ok arch linux is still there for you.


# Tech Industry Woes

What a year it has been. I can't say 2025 has been a great year in tech. It had a lot of ups and new releases but the industry and economy are not in a good spot. It seems like all the big tech companies are betting on spending enough money to create their AI god and make infinite money while the LLM still can't generate a proper quicksort algorithm.

```python
#from Opus 4.5:
#Me: "create quicksort in python"
#Slop:

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)
```

This isn't right since quicksort is an inplace algorithm and is not suppose to create extra memory with all those arrays. Now I know the AI bros will come out of the woodworks and begin to chant in their robes about me not prompting correctly. Sure the model can make a proper quicksort algorithm if you tell it to. In fact sometimes Opus 4.5 will give a longer response that looks like this

```python
Here's a classic quicksort implementation in Python:
pythondef quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)
Usage:
pythonnums = [3, 6, 8, 10, 1, 2, 1]
print(quicksort(nums))  # [1, 1, 2, 3, 6, 8, 10]
If you need an in-place version (more memory efficient):
pythondef quicksort_inplace(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pivot_idx = partition(arr, low, high)
        quicksort_inplace(arr, low, pivot_idx - 1)
        quicksort_inplace(arr, pivot_idx + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
Usage:
pythonnums = [3, 6, 8, 10, 1, 2, 1]
quicksort_inplace(nums)
print(nums)  # [1, 1, 2, 3, 6, 8, 10]
The first version is more readable; the second modifies the array directly without extra memory. Both have O(n log n) average time complexity.
```

This has problems too. The issue is that the LLM says you have two versions of quicksort to choose from. No, there is only one version of quicksort and it is an in-place algorithm. The first one does sort the array but it is not quicksort. No one is going to know that unless they have experience or understanding of the domain.

I don't care how many agents you chain together or how many planning agents are involved the results have to be reviewed very carefully and when the models drop 1000+ lines of code on you it becomes unfeasible. No your agent written test that pass don't even test what you think they are testing. This is the god that all the investors are banking on. The reason is that in a future where they think they can build a god and AGI/ASI then they can get infinite money. Sure if an AI could do everything a human can do but better then you can build robots and self improve everything to the nth degree. Keep in mind we are pretty behind since we were all suppose to have self driving cars now and if you go back to the 1980s we were suppose to be flying in all our vehicles.

Imagine if I told you that I spent 1 billion dollars and the service I sell you is riddled with issues and will produce false information that requires you to be a domain expert to sift through and get the most out of it. Imagine all that money spent and I still can't provide you a quicksort algorithm. Now in reality Anthropic has raised 27 billion dollars with a company evaluation of 350 billion...That seems very inflated for the type of service they deliver. AI companies are not valued by what they have they are valued on the premise that they can create something that will be superior in many ways to the current status quo and investors can get that sweet pile of money that awaits the first group to crack AGI/ASI.

## All these Layoffs are not due to AI

All the big companies this year had large layoffs. The headlines say we are moving towards being an AI first company or with AI our employees are so much more productive. This is not true and they know it. If you lay off engineers because you can replace them with an LLM then you are a short period away from having your products crash and burn. Look no further than Microsoft who boast [30% of code is written by AI](https://www.cnbc.com/2025/04/29/satya-nadella-says-as-much-as-30percent-of-microsoft-code-is-written-by-ai.html) (this is also not true) and has [windows 11 collapsing on some of it's core features](https://www.neowin.net/news/microsoft-finally-admits-almost-all-major-windows-11-core-features-are-broken/).

The other response is that employees with AI can produce 10x the work. This is another dream that investors want to be true. AI can speed some areas of development up but I think many times is a pseudo speed and you pay the price a few hours or days later. There are multiple studies on this. One of them by METR shows that [Surprisingly, we find that allowing AI actually
increases completion time by 19%—AI tooling slowed developers down.](https://arxiv.org/pdf/2507.09089) If AI really made everyone 10x more productive you may not need layoffs because you can have more people working at 10x and crank out products at a rate never seen before. This never happens and is a dream that others are banking on.

I can already hear the LLM fan boys saying that models have gotten better since the studies. This is true but they have not grown that much better. We can all fill the plateau. I know I have experienced this a lot with the newest models. I feel very fast at the beginning especially if it's a new project but then the sins begin to add up or I have to keep prompting to get what I want. In many cases if I would have done the work myself or just use a light layer of generated code I would have been much faster.

So if AI isn't the real reason then why are there all these layoffs? AI is a great excuse to pull the outsource trigger and trim the fat. Let's not disillusion ourselves. Big companies do not operate fast or efficiently. Some teams within them do but those are pretty rare. There are a lot of economic factors at play but this environment gives them a great excuse to get rid of teams that are not as critical and cut a some middle management layers. The darker angle is that these companies do layoffs then post jobs in other countries and try to bring the salaries for engineer/developer positions down. Outsourcing has been a thing for a long time and it get abused heavily in tech. At some point things will return to a more stable level but it is hard to predict how long this will last.

## Interviews Have Become Stupid

The candidate pool has become saturated from all these events. If you work in the USA you have to compete with all the people that got laid off along with the rest of the world that is competing for your job. If you are a new grad it is very hard since there are many experienced developers in the pool right now. It feels like devs are expected to know the world in order to write some web app or design a highly scalable flawless system at the drop of a hat. Don't get me wrong I actually enjoy some leetcode and I enjoy system design a lot. The issue is when you get shafted with an area you may not be strong in or know much about. It may be hard to design a signal analysis system if you have never worked with radars before. I have seen interviews where certain people are asked off the wall questions like prove this theorem just so there is an excuse not to hire them. These leetcode style interviews are bad filters in order to try and reduce the amount of people that big companies have to see.

There are many better interview tactics that let you know how well the person will work out technically. Drill down interviews work infinitely better in my experience than leetcode style. The issue with DSA rounds is you can have people perform well on them but fail at the job. Interviewing is difficult no matter the method. Drill downs allow people to pick a topic the person knows well and continue to go deep into the depths. This tells me exactly where they are at and their problem solving abilities. It is a great system for interviews. It is also very challenging and many may prefer DSA style interviews. Perhaps we can go deeper on this topic in another post

## Slop Has Infested Education

I teach some classes at a small university on the side. 
