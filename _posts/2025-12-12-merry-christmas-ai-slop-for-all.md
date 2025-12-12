---
layout: post
title: "Merry Christmas: Santa is Bringing AI Slop"
date: 2025-12-12
categories: ML
---

Ho Ho Ho to all vibe coders and all those failing production pipelines. Santa is coming to town and this year he's checking his AI generated naughty and nice list and didn't review it beforehand. His fat self is sliding down your dirty chimney and your present this year is a toy that looks correct at a glance but the more you look at it you realize Santa just gave you Mrs. Claus's faulty elf action figure with 12 fingers and a message attached that says "LGTM I pushed this +500 changes through to prod. My agent workflow told me it was high quality". 

Santa climbs his way to the roof and heads off but instead of doing his normal route that uses `A*` search and plans an optimal path he is using a new sleek agent workflow that charted his path. The agents said he was absolutely right about this being better than old crusty `A*`. The only issue is it cost thousands of dollars to make this new route and the agent workflow had a hiccup and kept saying Santa isn't real and refused to make a route for him to follow. 

Now due to this new expensive route planning tool he had to layoff some elves and he is now vibing his way through your neighborhoods with no plan. Your fake girlfriend doll is never coming but it's ok arch linux is still there for you.


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

This isn't right since quicksort is an inplace algorithm and is not supposed to create extra memory with all those arrays. Now I know the AI bros will come out of the woodworks and begin to chant in their robes about me not prompting correctly. Sure the model can make a proper quicksort algorithm if you tell it to. In fact sometimes Opus 4.5 will give a longer response that looks like this

```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)
```
```
Usage:
pythonnums = [3, 6, 8, 10, 1, 2, 1]
print(quicksort(nums))  # [1, 1, 2, 3, 6, 8, 10]
If you need an in-place version (more memory efficient):
```

```python
def quicksort_inplace(arr, low=0, high=None):
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
```

```
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

All the big companies this year had large layoffs. The headlines say we are moving towards being an AI first company or with AI our employees are so much more productive. This is not true and they know it. If you lay off engineers because you can replace them with an LLM then you are a short period away from having your products crash and burn. Look no further than Microsoft who boast [30% of code is written by AI](https://www.cnbc.com/2025/04/29/satya-nadella-says-as-much-as-30percent-of-microsoft-code-is-written-by-ai.html) (this is also not true) and has [windows 11 collapsing on some of its core features](https://www.neowin.net/news/microsoft-finally-admits-almost-all-major-windows-11-core-features-are-broken/).

The other response is that employees with AI can produce 10x the work. This is another dream that investors want to be true. AI can speed some areas of development up but I think many times is a pseudo speed and you pay the price a few hours or days later. There are multiple studies on this. One of them by METR shows that [Surprisingly, we find that allowing AI actually
increases completion time by 19%—AI tooling slowed developers down.](https://arxiv.org/pdf/2507.09089) If AI really made everyone 10x more productive you may not need layoffs because you can have more people working at 10x and crank out products at a rate never seen before. This never happens and is a dream that others are banking on.

I can already hear the LLM fan boys saying that models have gotten better since the studies. This is true but they have not grown that much better. We can all feel the plateau. I know I have experienced this a lot with the newest models. I feel very fast at the beginning especially if it's a new project but then the sins begin to add up or I have to keep prompting to get what I want. In many cases if I would have done the work myself or just use a light layer of generated code I would have been much faster.

So if AI isn't the real reason then why are there all these layoffs? AI is a great excuse to pull the outsource trigger and trim the fat. Let's not disillusion ourselves. Big companies do not operate fast or efficiently. Some teams within them do but those are pretty rare. There are a lot of economic factors at play but this environment gives them a great excuse to get rid of teams that are not as critical and cut some middle management layers. 

The darker angle is that these companies do layoffs then post jobs in other countries and try to bring the salaries for engineer/developer positions down. Outsourcing has been a thing for a long time and it gets abused heavily in tech. At some point things will return to a more stable level but it is hard to predict how long this will last.

## Interviews Have Become Stupid

The candidate pool has become saturated from all these events. If you work in the USA you have to compete with all the people that got laid off along with the rest of the world that is competing for your job. If you are a new grad it is very hard since there are many experienced developers in the pool right now. It feels like devs are expected to know the world in order to write some web app or design a highly scalable flawless system at the drop of a hat. 

Don't get me wrong I actually enjoy some leetcode and I enjoy system design a lot. The issue is when you get shafted with an area you may not be strong in or know much about. It may be hard to design a signal analysis system if you have never worked with radars before. I have seen interviews where certain people are asked off the wall questions like prove this theorem just so there is an excuse not to hire them. These leetcode style interviews are bad filters in order to try and reduce the amount of people that big companies have to see.

There are many better interview tactics that let you know how well the person will work out technically. Drill down interviews work infinitely better in my experience than leetcode style. 

The issue with DSA rounds is you can have people perform well on them but you gain no extra insight into how they will do at the job. Interviewing is difficult no matter the method. Drill downs allow people to pick a topic the person knows well and continue to go deep into the depths. 

This tells me exactly where they are at and their problem solving abilities. It is a great system for interviews. It is also very challenging and many may prefer DSA style interviews. Perhaps we can go deeper on this topic in another post.

## Slop Has Infested Education

I teach some classes at a small university on the side. Pack it up ladies and gentlemen it's over. I don't even blame students. Some courses are just doomed. The good news is Computer Science is not doomed but the people in it may be. I have found raw `HTML` in `Dockerfiles`. I have seen students go into great details on how a transformer works but can't write a simple `GET` request using `FASTAPI`. Students will generate code that does not solve the problem or it only does 10% of the actual work needed.

One of the issues is the students have offloaded their thinking to an LLM and do not know what to look for or prompt for. In the past if students didn't know how to do something they just struggled to find some code online to copy or copy from their classmates. That's not as bad since they have to put in effort to find what they need. Now they get an instant garbage answer from an LLM.

You can't fight this either. You can't ban it. LLMs are here to stay. The way to go about this when designing your CS courses is to have the courses be more engaging and thoughtful. LLMs mean you can't assign certain types of homework or have any writing assignments that can easily be done with LLMs.

The real problem is that students seem to get by with LLM generated answers or heavy assistance from them. I understand programming 101 cannot be that hard but you almost have to account for LLM help in all assignments. Now in my experience the students will not have the knowledge to ask the right questions so most of your programming assignments beyond "write a for loop" or "write a class" will have some interesting answers to say the least. 

You would think LLMs could do basic programming homework but many times students ask wild questions that lead to crazy code snippets. Your class should be more open in the problems that it solves (more end-to-end solutions) and exams should be done one-on-one if possible. The way to tackle this issue is to make sure even with LLM assistance students still have to dig deep and understand the concepts. You cannot be a lazy professor and yes this means you have to redo your curriculums.

Students, you need to lock in and brace yourself for absolute chaos. Your grades do not matter but your knowledge does. You can get an A and be an absolute idiot. Forget the grades and understand the theory and execution. Your grades will be fine if you do this. You need to be asking your professor why and how all this works. Ask and engage with the material. If there is a chance to work on a real or semi real project do it. Start a project and ask the teacher for input or code review. The time of just passing by is over. You need to be alert and get your head in the game.

I will point out that after teaching I am not worried at all about job security. Devs who went to school and learned the fundamentals may be the last people who know how a computer works. The jobs may all suck but there will be so many "Fix our companies broken apps" jobs that we will never be out of work. The issue is now for 40+ hours a week you are debugging LLM code that no one can tell you how anything works. The amount of CVEs will be exponential and legacy code will be any code produced last month by Claude/GPT/Gemini at a rate that makes you want to start farming and raising chickens.

## Consumer Markets are Cooked

I am not even going to address the GPU prices and supply issues. We all know those are issues. Now the AI grim reaper has destroyed the ability to buy DDR5 RAM. RAM...The part of your PC that you could always buy for cheap has in many cases overshot the prices of CPUs and some GPUs. The 64 GB RAM kit I bought one year ago for ~$150 is now sitting around ~$755. How is this even allowed? What world am I living in where RAM is more than my CPU? 

No one can afford to build a PC now with all these past years of shenanigans. This is the final straw for a lot of people. I don't know what to tell people who want to build an affordable PC. The reason for all of this is so resources can go to fund data centers and frontier model development. You can't afford anything because we need to send these valuable resources to OpenAI so we can have a model hallucinate (this term used to be called being wrong) about how to sort an array. 

My fear is that this drives people into being forced to rent their computers from cloud services or people get marketed to move everything to the cloud where you can rent 64GB of RAM. What an awful world to live in. You can hate AI all you want but you are being affected either way. You will own nothing and be happy.

## Awful AI Integration

Every product now comes with a broken LLM when there is no need for one. Even Pytorch website and Unreal website has them and they suck. It provides no value and is borderline broken. I would be more productive manually searching the docs every time. Leetcode has another awful chatbot to "help" you on problems and it flat out does not work. It just makes the experience and software worse. Windows keeps wanting to shove Copilot down my throat and I'm sick of it. No one wants to use these things. Remove them and focus where people need you to focus.

## Open Source in a Bind

I feel for all OSS maintainers. You guys are getting pelted with awful PRs and odd "fixes" that do not fix anything. LLMs are scanning codebases and just submitting bug reports with little or no supervision. The most famous one being the [curl buffer overflow](https://hackerone.com/reports/2823554) that does not exist. This is a problem when you flood the community with bug reports that have not been verified.

OSS communities are hard pressed for time and under appreciated. They already get hate from people demanding fixes or features. They don't need AI slop added to their list of items to address.

# Pushing Through the Slop

How do we navigate going into 2026? Well unfortunately I think it is going to get worse. AI will be pushed harder than ever. People will repeat the doom cycle every small LLM release even if there are no major improvements. The Big Wigs will continue to froth at the mouth demanding their AGI god will be complete if we can raise another round of billions. So where does that leave us? I think when it comes to software engineers there is still plenty of demand. The threat as of today is not AI it's people. People who want to bring salaries down, outsource, and attempt to replace humans with terrible expensive LLMs. So here are some things to consider

## Bet on yourself

Despite what is happening in the world the need for good engineers is very high. LLMs are changing some workflows and can be used in productive ways but there is no substitute for having deep knowledge. LLMs in some ways force you to go deeper. You are going nowhere if all you can do is wire up some basic CRUD api and that is where your knowledge stops. Since many who read this article are ML engineers if all you can do is train a classifier or inference a pre-trained model then you too are going into the dump. 

If your skills can easily be prompted then you have no skills. I don't mean the code either. With the right prompts you can get some decent output. What I mean is if you can't think in a more complete way or know how to design solutions for hard problems then you might want to get busy studying. If you only know python and can't optimize just a little for a faster training loop or inference then we have problems. Always be hungry to grow your abilities. Even at the higher level if your soft skills suck (they probably do) then you need to sharpen those up if you want to have a successful project.

Never outsource your brain to a LLM always bet on yourself and the ability to get better.

## Detach from Job Titles

I don't care what position or what company you work for it can all vanish in one day. Your fancy senior or principal title won't mean much when you get laid off. One day you may be walking into Google for a free lunch and the next day they kick your typescript kiddie butt out of there.

The point is you should not make your job title the way you find fulfillment. It is out of your control in many cases. Keep growing your skills and becoming better. If you lose your job you still have your skills. You can perform very well and still be laid off for no good reason. You are not a bad dev (unless you are). All those rejection interviews and failed DSA interviews when you were a bigshot can be disappointing. You may be used to being respected but now you are competing with the whole world to find your next job. Don't get caught up in needing a fancy title.

Also find joy in working on something outside of work. Many if not most developers work jobs that they don't enjoy. Start a project that you can have fun with. Even those big tech jobs are not all they crack up to be. Perhaps you would be better off at a smaller company or be independent.o work on lower level problems like compilers and language design you will need to get deep into `C++`

## Find Fields that are Difficult

Find a field that is hard and begin to understand how to work effectively in it. Look no domain specialist is talking about LLMs taking their jobs. I don't hear a lot of compiler, ML, quant, geo-spatial engineers pondering their job landscape. The people who seem to be in fear are noobs (rightfully so) and people who are lower skilled. It's just time to dig deeper and develop in areas that require more effort.

Here are some ideas of skills to work on if you are looking into some of these areas

### Compilers and Language Design

* `C++`
* JIT vs AOT
* How Interpreters work
* Abstract Syntax trees
* Efficient token parsing

### ML

* Calculus and Linear algebra
* How a GPU works
* Python and CUDA
* Inference Strategies (graph optimization, test time scaling, etc...)

### Quant

* `C++`
* Trading Strategies and Timelines
* Server locations
* Memory Optimizations
* CPU caching

### Geo-Spatial

* Coordinate Systems (ECEF, WGS-84, etc...)
* Resolution of images
* Time Zones
* Rotations and NODATA

These are all valuable skills in dire need.

## Search for Niche Markets

Look that major tech VC won't care about an old crusty invention like a inventory tracker or optimized databases but that auto dealer or lumber yard might praise your name for ages. If you do independent work or small team work you can capture an unreached or under appreciated market. Maybe that small business is growing but can't afford a full time dev team to figure out why their database is slow. Maybe their mobile app is broke or website can't work on smaller screens. You can provide great value to them and honestly it's more value than your bigshot title and a company of thousands where you are nothing but a cog in the machine.

Another prospect is doing the dirty jobs. Get good at `COLBOL` or `ADA`. Everyone who knew those is six feet under. You could walk into the auto insurance industry and name your price. Many banks still have `FORTRAN` code. They need developers more than anyone. Learn those and profit.

Be on the lookout for the next big market known as a vibe fixer. Someone needs to fix that month old legacy code.

## Advice to Students and Beginners

Do not believe the marketing talk about AI taking over software development. This field is a difficult one and it will continue to be difficult. Please get curious about programming. Become obsessed over something and keep exploring those rabbit holes. It may seem like an LLM knows more than you but it does not. You can surpass its regurgitating text in few months and you can even begin to know how to use these tools to best suit your development flow.

It may seem like a never ending uphill battle but you can overcome it. You wouldn't substitute your doctor or surgeon for an LLM just like you wouldn't substitute your engineering team with LLMs.

LLMs can be helpful for learning if you use them to help explain or make exercises for you to learn from. They can also make visuals and charts to explain concepts. Next time you struggle on a subject ask for an LLM to explain it and diagram the process. As always you need to verify the output.

It's like any other field, an LLM is better than me at biology questions but someone who spent a one to three years working in the field of biology would be able to use the LLM more effectively and know when its wrong.

It will take time but you can surpass the LLM. LLMs are very bad at making extendable or testable systems. This makes sense since most people struggle with this and most code on the internet is pretty bad. You can learn to make better systems than an LLM in under a year. You can for sure learn how to use an LLM and know when its wrong in under a year.

If you are new to the field you need to ignore the doom and gloom and buckle down. There is no way around working hard. I promise you can study and become much better than an LLM and be very useful for others.

Here are some concrete skills to learn for a few major areas of development

### Frontend

* How Browsers render html
* How the DOM works
* Server Side vs Client Side
* Tailwind vs Standard CSS
* How to separate concerns in your component design
* Managing State

### Backend

* REST vs Web Sockets vs GRPC
* Inter-process communication
* Database Access
* Error handling
* Good endpoint design

### Deployment

* Optimizing Docker Images
* Docker Caching and Layers
* Reverse Proxies
* Gitlab/Github workflows

These are all areas you can go deep in and if you can demonstrate some of these you will be more likely to get hired

## Sanity Will Return so be a Cockroach

At some point (not next year) sanity will come back. The market will bounce back and developers will have a more favorable market. The trick is to survive the great culling and stick around. The way to survive is to adapt and continue to be in demand. It may look different if all the markets are saturated and you may need to adjust expectations. 

If you are used to only working in large companies like fortune 500 or big tech and they are all not hiring or outsourcing large amounts of roles it may be time to go to the medium or smaller size companies. At the end of the day you have to eat. Remember it's not about who will show up it's about who's left. Push through these hard times and eventually you will make it through and be in a better position.

# Conclusion

This whole article was generated by AI. Just kidding I wouldn't do that to you all. If this article sucks then it's because of human slop. I am always going to bet on the human. We will get through these hard times together. It will be painful and we are at the bottom of this trickle down AI slop fest. I encourage us all to dig deeper and find the things in life we can improve upon. Merry Christmas to all and I hope Santa stops the slop.

