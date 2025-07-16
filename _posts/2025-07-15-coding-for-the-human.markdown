---
layout: post
title: "Oh the Humanity"
date: 2025-07-15 18:30:00 -0500
categories: tech
---

You just got done sprint planning and you feel depressed. The boss needs this new feature cranked out on the pronto so you get busy. You get your coffee and open up your editor only to be confused on where to begin coding. You thought this system would be quick to figure out for that the team knew rouphly where to begin but alas it's up to you figure it all out again. This is the 900th maze you have had to navigate this year.

Every line is an abraction and you are now 7 layers deep into what was one simple function call. You are lost in the sauce and no one is coming to save you. Not even the lead dev understands where you are. You panic and run to the `README.md` but nothing. the `README.md` ironically has nothing to read. Nothing at all on how to spin up the system or debug it. You spend hours upt to days just figuring out how current events work. You beg for relief from your LLM but it's spitting nonsense and the context length is too long. You start asking around for developers who have been working around this codebase. Turns out everyone around you just laughs and says "yeah, this codebase is a mess" then they trail off. No one is taking the time to fix things or document the project. After a long week if your lucky you get your feature done but you lost a part of your soul in the process.

This is what we are going to go over today. How not be a rude moron to your fellow devs and give your team the ability to succeed (even if it's just a small increase in developer productivity)

## Human Centric Problem Solving

You need to get your head out of your own butthole and ask yourself if a fellow dev comes after me will they be better equiped than myself to edit or change things. Will they even know how to launch the service locally and debug? Do they know how to run the test that are written (if there are no test then sorry mate push to prod and see what happens)? It's fine that your problem solving skills allowed you to get the feature working but there are more problems to solve. You are not done yet. I do understand crazy deadlines so don't think I am ignoring that component but you can always sneak in at least some docs or something while you wait on the MR to be reviewed.

What is human centric problem solving? why are we talking about the human when we are in the machine. Well the human is important even if you are some fat lonely person with anime waifus in your terminal. There is an outside world and I'm living in it so try to be respectful and throw me a bone while I traverse this awful codebase. Human centric problem solving is shifting perspective and thinking how does the way this problem I just solved effect my co-workers? Is this code heavily abstracted and hard to find the meat? Do I have test that let others know that the code is working how I expect it to. Did I scatter my solution over 10 files and now they have to bunny hop around this code like the easter bunny? Are their docstrings and user guides on how to use the functions or program I just made. Solving problems goes beyond just the code and getting it to work, it's also how it affects the team. 

I understand that in legacy systems this is an uphill battle and you are going to bleed during it. This is extremely hard to do the longer the project goes on and if the team has had high turnover in the past years. You can still apply what we are talking about to legacy code basis but it is a much slower grind to get there.

If you are leading smaller projects and have full control or large amounts of control then this is an excellent oppertunity to try to create a system that enables others to work. You do not need to be a caveman and do production debugging.

## Code does WHAT?

Let's look at this example of sanitizing a username and email

```python
def clean_username_and_email(username, email):
    if not username or '@' not in email:
        return False
    
    clean_username = ""
    for char in username:
        ascii_val = ord(char)
        if (ascii_val >= 65 and ascii_val <= 90) or \
           (ascii_val >= 97 and ascii_val <= 122) or \
           (ascii_val >= 48 and ascii_val <= 57):
            clean_username += char
    
    clean_email = email.strip().lower()
    
    user_id = 0
    for i, char in enumerate(clean_username):
        ascii_val = ord(char)
        multiplier = 1
        for _ in range(i + 1):
            multiplier *= 31
        user_id += ascii_val * multiplier
        user_id = user_id % 100000
    
    user = {
        'id': user_id,
        'username': clean_username,
        'email': clean_email
    }
    return user
```

### Roller Coaster Tycoon is Back

There are several issues with the way this problem was solved but for the purpose of this article we are going to talk about the flow of the program and how it sends the developer on a roller coaster ride. The issue is the flow of this code goes from high level operations, then to low level operations, then back to high level, and so on. it's like a bad girlfriend for those developers that know what a female is.

```python
if not username or '@' not in email:
        return False
```

This is easy to follow and we all know what is going on. Sure there are other issues but as far as what this code is doing it is clear. Then we swoop down to ascii land like a kamikaze.

```python
clean_username = ""
for char in username:
    ascii_val = ord(char)
    if (ascii_val >= 65 and ascii_val <= 90) or \
        (ascii_val >= 97 and ascii_val <= 122) or \
        (ascii_val >= 48 and ascii_val <= 57):
        clean_username += char
```

This is such a shift in flow. I now have to stop and think why are they doing this. Is there a reason for this madness or did they lost their minds and choose to do the hardest most mysterious thing. It is not clear what this code is doing at first glance. TLDR: this code is only accepting alphanumeric characters and appending them to `clean_username`. But here is the problem **NO ONE MEMORIZES ASCII CODE** go up to your co-workers and ask them what ascii value 65 is. Unless you love memorizing numbers no one will know that and honestly you deserve to be punched in the face for asking me.

So let's clean this up a bit. A pythonic way of doing this would be to create a function that cleans the username or if you prefer to not have the function you can just incorporate the single line

```python
def remove_nonalphanumeric_characters(username):
    """Remove non-alphanumeric characters from username."""
    return ''.join(char for char in username if char.isalnum())
```

Here is something that you may find more often

```python
import re

def sanitize_username(username):
    """Keep only alphanumeric characters."""
    return re.sub(r'[^a-zA-Z0-9]', '', username)
```

This is using a regular expression to filter the username. While no one can read ascii a regex is a close 2nd. However people are more inclined to know regex than ascii and it it pretty easy to look up and see what a regex expression is doing. Both of these approaches are easier to follow. No longer do you have to go down to the lower levels of ascii filtering.

Then the next line takes up back up to a high level operation

```python
clean_email = email.strip().lower()
```

So you just continue this roller coaster ride until you crash out and go home. I'm sure you get my point of the flow of the program and how it reads to an outsider. You should try to ease others into the flow of your program when possible.

### Begin the Cleaning

Let's ask our brain dead friend Claude to clean this code up.

```python
import re
import hashlib
from typing import Optional, Dict, Any


class UserValidationError(Exception):
    """Raised when user input validation fails."""
    pass


def clean_username_and_email(username: str, email: str) -> Dict[str, Any]:
    """
    Adds a user to the database after validation and sanitization.
    
    Args:
        username: Raw username input
        email: Raw email input
        
    Returns:
        Dictionary containing user data with id, username, and email
        
    Raises:
        UserValidationError: If input validation fails
    """
    # Step 1: Validate inputs
    _validate_user_input(username, email)
    
    # Step 2: Sanitize data
    clean_username = _sanitize_username(username)
    clean_email = _sanitize_email(email)
    
    # Step 3: Generate user ID
    user_id = _generate_user_id(clean_username)
    
    # Step 4: Create user record
    user = {
        'id': user_id,
        'username': clean_username,
        'email': clean_email
    }
    
    return user


def _validate_user_input(username: str, email: str) -> None:
    """
    Validates that username and email meet basic requirements.
    
    Raises:
        UserValidationError: If validation fails
    """
    if not username:
        raise UserValidationError("Username cannot be empty")
    
    if not email or '@' not in email:
        raise UserValidationError("Email must contain @ symbol")


def _sanitize_username(username: str) -> str:
    """
    Removes non-alphanumeric characters from username.
    
    Args:
        username: Raw username input
        
    Returns:
        Username containing only letters and numbers
    """
    return re.sub(r'[^a-zA-Z0-9]', '', username)


def _sanitize_email(email: str) -> str:
    """
    Normalizes email to lowercase and removes whitespace.
    
    Args:
        email: Raw email input
        
    Returns:
        Normalized email address
    """
    return email.strip().lower()


def _generate_user_id(username: str) -> int:
    """
    Generates a consistent user ID based on username.
    
    Uses SHA-256 hash for better distribution and collision resistance
    compared to the original manual hash function.
    
    Args:
        username: Clean username
        
    Returns:
        Integer user ID between 0 and 99999
    """
    # Create hash of username
    hash_object = hashlib.sha256(username.encode())
    hash_hex = hash_object.hexdigest()
    
    # Convert first 8 characters to integer and mod to keep manageable
    hash_int = int(hash_hex[:8], 16)
    return hash_int % 100000
```

Let's re-examine the control flow. I think this is better but now it suffers from the religious cult of clean code where functions have to be small and only do one thing. One saving grace is that at least the locality of scope is ok and these functions are not in different files.

```python
# Step 1: Validate inputs
_validate_user_input(username, email)
```

Now I have to jump down to another function where all it's doing is two conditionals in order to throw an error. Just save me the trouble and add those checks to the top level function. I dont really need to abstract this. In a real more complicated sanitation function you will need to do more and would be justified to make it into a seperate function to help make testing easier. In this specific example it is a bit annoying. I am being a bit pandantic but I want you to be aware of how often other people have to hop around to figure out the abstractions.

Especailly for this. I have seen this a lot.

```python
def _sanitize_email(email: str) -> str:
    """
    Normalizes email to lowercase and removes whitespace.
    
    Args:
        email: Raw email input
        
    Returns:
        Normalized email address
    """
    return email.strip().lower()
```

did you just wrap a standard string operation in your own function? Are you trying to make an abstraction over the standard string library abstraction? This is just ridiculous. You are going to make me peek or hop into a function where all you did was create a useless wrapper over a perfectly fine standard string call. I dont need more code to review or more docstrings to maintain. Get this crap out of here. Just call the line of code in the top level function you dummy.


### Less Abstract Art

Just cut the crap out and make the function transparent and clear

```python
def clean_username_and_email(username, email):
    if not username:
        raise UserValidationError("Username cannot be empty")
    
    if not email or '@' not in email:
        raise UserValidationError("Email must contain @ symbol")
    
    cleaned_username = ''.join(char for char in username if char.isalnum())

    cleaned_email = email.strip().lower()

    user_id = _generate_user_id(clean_username)
    
    return {
        'id': user_id,
        'username': clean_username,
        'email': clean_email
    }
```

We can break things out when we need to but do not create endless abstractions becuase you might need them in the future. Be direct and code with your chest out. The only function worth keeping from a flow of the program perspective in the `_generate_user_id` since that does do more things and is easier to test when isolated. Please keep in mind that this is not a good way to get a user id. There are many flaws in this whole program but today we mainly talked about the flow as a developer reading the code.
