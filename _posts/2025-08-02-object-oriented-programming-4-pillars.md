---
layout: post
title: "Object Oriented Programming: The 4 Flimsy Pillars"
date: 2025-08-02
categories: ML
---

This will be a quick dive into how everything these ivory tower professors taught you is false about OOP. Most professors in CS departments beat it into your head that these 4 pillars of OOP opened the third eye of programmers and allowed for programming to change and become the way to design systems and solve problems.

Let's go over to 4 Pillars:

* Encapsulation
* Inheritance
* Polymorphism
* Abstraction

# Encapsulation

The formal definition according to wikipedia is 

>
>encapsulation refers to the bundling of data with the mechanisms or methods that operate on the data. It may also refer to the limiting of direct access to some of that data, such as an object's components. Essentially, encapsulation prevents external code from being concerned with the internal workings of an object.
>

Boiled down encapsulation is how you bundle your data or operations to a system to a concise interface and hide away parts you do not want exposed. Below is a code example in python which is probably the worse language to represent this.

```python
class BankAccount:
    def __init__(self, owner, initial_balance=0):
        self.owner = owner
        self.__balance = initial_balance  # Private attribute (name mangling)
        self._account_number = "ACC123456"  # Protected attribute (convention)
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return f"Deposited ${amount}. New balance: ${self.__balance}"
        return "Invalid deposit amount"
    
    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            return f"Withdrew ${amount}. New balance: ${self.__balance}"
        return "Insufficient funds or invalid amount"
    
    def get_balance(self):  # Public method to access private data
        return self.__balance

# Usage
account = BankAccount("Alice", 1000)
print(account.deposit(500))
print(account.withdraw(200))
print(f"Balance: ${account.get_balance()}")
```

Here is the same example but in an actual language meant for adults

```c++
class BankAccount {
private:
    std::string owner;
    double balance;           // Private data member
    std::string accountNumber; // Private data member

public:
    // Constructor
    BankAccount(const std::string& ownerName, double initialBalance = 0.0) 
        : owner(ownerName), balance(initialBalance), accountNumber("ACC123456") {}
    
    // Public interface methods
    std::string deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            return "Deposited $" + std::to_string(amount) + 
                   ". New balance: $" + std::to_string(balance);
        }
        return "Invalid deposit amount";
    }
    
    std::string withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return "Withdrew $" + std::to_string(amount) + 
                   ". New balance: $" + std::to_string(balance);
        }
        return "Insufficient funds or invalid amount";
    }
    
    // Getter method for private data
    double getBalance() const {
        return balance;
    }
    
    std::string getOwner() const {
        return owner;
    }
};
```

The problem with being taught this in school is most professors champion this as something unique to OOP or that OOP made encapsulation better.

OOP really just moves the encapsulation around. One of the most annoying things is the private/public member with all the getters and setters. If most of your members or all of your members are private and you have getters/setters

```python
class Person:
    def __init__(self):
        self.__name = ""
    
    def get_name(self): return self.__name
    def set_name(self, name): self.__name = name

vs

class Person:
    def __init__(self):
        self.name = ""
```

This is crazy to me. This creates extra memory and stupid abstractions just to modify or retrieve a member that should be public. This happens all the time in the wild.

Here are some non OOP ways to handle encapsulation.

## Use Yo Struct

Python does not have structs in the traditional sense but in a real language it may look like

```c
// Instead of hiding data in objects
struct Player {
    float x, y, z;
    float health;
    int weapon_id;
};

// Functions operate on the Struct
void update_player_position(struct Player* players, int count, float dt);
void apply_damage(struct Player* player, float damage);
```

python has dataclasses and with slotting them you can kinda have a struct and save a bit of memory.

```python
@dataclass(slots=True)
class Point2D:
    x: float
    y: float
    z: float
    health: float
    weapon_id: int
```

You could also used NamedTuples if you want immutable data. This way you don't have to deal with a whole object and the public/private issues that can arise. You also can take a more functional approach of operating on the data if you want.


## C Did it Better

Before the days of OOP, The C language had stronger encapsulation than a lot of OOP languages.

```c
// header file
#ifndef FILE_H
#define FILE_H

#include <stddef.h>

// Forward declaration - compiler knows File exists but not its contents
typedef struct File File;

// Public interface - users can ONLY use these functions
File* file_create(const char* filename);
int file_write(File* file, const char* data);
char* file_get_filename(File* file);  // Controlled access to filename
void file_destroy(File* file);

#endif
```

```c
// implementation file
#include "file.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The ACTUAL struct definition - ONLY visible in this file
struct File {
    FILE* handle;           
    char* filename;         
    int is_open;          
    size_t bytes_written; 
    int magic_number;
};

File* file_create(const char* filename) {
    File* file = malloc(sizeof(struct File));
    if (!file) return NULL;
    
    file->filename = strdup(filename);
    file->handle = fopen(filename, "w+");
    file->is_open = (file->handle != NULL);
    file->bytes_written = 0;
    file->magic_number = 0xDEADBEEF;
    
    return file;
}

int file_write(File* file, const char* data) {
    if (!file || !file->is_open) return -1;
    
    size_t len = strlen(data);
    size_t written = fwrite(data, 1, len, file->handle);
    file->bytes_written += written;
    
    return (written == len) ? 0 : -1;
}

// Controlled access - we decide what users can see
char* file_get_filename(File* file) {
    if (!file) return NULL;
    return file->filename;
}

void file_destroy(File* file) {
    if (file) {
        if (file->handle) fclose(file->handle);
        free(file->filename);
        free(file);
    }
}

```

The user can only use what is in the header file and the program would crash if the user tried to reference internal `File` members.

In my mind this is a much stronger enforcement of the encapsulation pillar and it's in C. There ain't no objects here

# Inheritance

This was just a bad idea. Every time people reach for this tool it bites them back two fold. I would say that many people try to not use inheritance or if forced to they try not to go more than one layer deep. Honestly, this deserves its own rant but lets just hit on why this sucks.

## The Abusive Parent

If you need to change the base class which happens more than it should in the real world then prepare to get beaten. You force all the other implementation classes to carry whatever bloat you made in the base class.

You also end up in confused pretty quickly. You have to keep hopping down the trail into the nested class structure and most people when they get 3+ layers deep their brain gives up trying to solve the murder mystery.

```python
class Rectangle:
    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height
    
    def set_width(self, width: float):
        self._width = width
    
    def set_height(self, height: float):
        self._height = height
    
    def get_width(self) -> float:
        return self._width
    
    def get_height(self) -> float:
        return self._height
    
    def area(self) -> float:
        return self._width * self._height

class Square(Rectangle):
    def __init__(self, side: float):
        super().__init__(side, side)
    
    def set_width(self, width: float):
        self._width = width
        self._height = width
    
    def set_height(self, height: float):
        self._width = height
        self._height = height
```

In this stupid example the square class is forced to hold onto the height and width attribute of the parent. It has to manage these resources. A square does not need a width and a height to maintain. It can be the shared value since width and height are equal in a square but now you get to make sure they are always equal.

## No more Abuse

Other ways to handle this situation could be composition, traits, entity componenet system, etc...

```python
@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0

@dataclass
class Velocity:
    dx: float = 0.0
    dy: float = 0.0

@dataclass
class Health:
    current: int = 100
    maximum: int = 100

@dataclass
class Render:
    sprite: str = "?"
    color: str = "white"

@dataclass
class Player:
    name: str = "Unknown"

@dataclass
class Enemy:
    damage: int = 10

@dataclass
class Powerup:
    effect: str = "heal"
    value: int = 20

# Entity is just a container for components
class Entity:
    def __init__(self, entity_id: int):
        self.id = entity_id
        self.components: Dict[Type, Any] = {}
    
    def add_component(self, component: Any) -> 'Entity':
        self.components[type(component)] = component
        return self
    
    def get_component(self, component_type: Type):
        return self.components.get(component_type)
    
    def has_component(self, component_type: Type) -> bool:
        return component_type in self.components
    
    def remove_component(self, component_type: Type):
        if component_type in self.components:
            del self.components[component_type]

# World manages all entities
class World:
    def __init__(self):
        self.entities: Dict[int, Entity] = {}
        self.next_id = 1
    
    def create_entity(self) -> Entity:
        entity = Entity(self.next_id)
        self.entities[self.next_id] = entity
        self.next_id += 1
        return entity
    
    def get_entities_with_components(self, *component_types: Type) -> List[Entity]:
        result = []
        for entity in self.entities.values():
            if all(entity.has_component(comp_type) for comp_type in component_types):
                result.append(entity)
        return result
    
    def remove_entity(self, entity_id: int):
        if entity_id in self.entities:
            del self.entities[entity_id]

# Systems operate on entities with specific components
class MovementSystem:
    @staticmethod
    def update(world: World, delta_time: float):
        # Find all entities with both Position and Velocity
        moving_entities = world.get_entities_with_components(Position, Velocity)
        
        for entity in moving_entities:
            pos = entity.get_component(Position)
            vel = entity.get_component(Velocity)
            
            # Update position based on velocity
            pos.x += vel.dx * delta_time
            pos.y += vel.dy * delta_time

```

In this pattern you have a container that holds all the entities and then you have systems that query and operate on the entities held.

You could use composition

```python
class KiBlast:
    def fire(self):
        return "💥 KAMEHAMEHA! Powerful ki blast fired!"
    
    def charge_up(self):
        return "⚡ Charging ki energy..."

class FlightAbility:
    def fly(self):
        return "🚀 Flying at super speed!"
    
    def land(self):
        return "Landing gracefully"

class Transformation:
    def __init__(self, form_name, power_multiplier):
        self.form_name = form_name
        self.power_multiplier = power_multiplier
        self.active = False
    
    def transform(self):
        if not self.active:
            self.active = True
            return f"⚡ TRANSFORMING TO {self.form_name.upper()}! Power x{self.power_multiplier}!"
        return f"Already in {self.form_name} form"
    
    def power_down(self):
        if self.active:
            self.active = False
            return f"Powering down from {self.form_name}"
        return "Not transformed"

class Scouter:
    def scan(self, target_name):
        power_level = 9000  # It's over 9000!
        return f"{target_name}'s power level is {power_level}!"

# Fighter uses composition - they HAVE these abilities
class Fighter:
    def __init__(self, name):
        self.name = name
        self.power_level = 1000
        
        # Optional abilities - start as None
        self.ki_blast = None
        self.flight = None
        self.transformation = None
        self.scouter = None
    
    def learn_ki_blast(self):
        self.ki_blast = KiBlast()
        return f"{self.name} learned ki blast techniques!"
    
    def learn_flight(self):
        self.flight = FlightAbility()
        return f"{self.name} learned how to fly!"
    
    def gain_transformation(self, form_name, multiplier):
        self.transformation = Transformation(form_name, multiplier)
        return f"{self.name} gained {form_name} transformation!"
    
    def get_scouter(self):
        self.scouter = Scouter()
        return f"{self.name} equipped a scouter!"
    
    def attack(self):
        if self.ki_blast:
            return self.ki_blast.fire()
        return f"{self.name} throws a punch!"
    
    def fly_around(self):
        if self.flight:
            return self.flight.fly()
        return f"{self.name} can't fly yet - running instead!"
    
    def power_up(self):
        if self.transformation:
            return self.transformation.transform()
        return f"{self.name} powers up normally"
    
    def scan_enemy(self, enemy_name):
        if self.scouter:
            return self.scouter.scan(enemy_name)
        return f"{self.name} can't scan power levels without a scouter"

# Create different fighters with different abilities
print("Creating Dragon Ball Z fighters:")

# Weak fighter - just basic abilities
krillin = Fighter("Krillin")
print(f"🧑 {krillin.name} created")
print(f"{krillin.name}: {krillin.attack()}")  # Just punches
print(f"{krillin.name}: {krillin.fly_around()}")  # Can't fly yet

print()

# Train Krillin
print(f"💪 {krillin.name} training:")
print(krillin.learn_ki_blast())
print(krillin.learn_flight())
print(f"{krillin.name}: {krillin.attack()}")  # Now has ki blast!
print(f"{krillin.name}: {krillin.fly_around()}")  # Now can fly!

print()

# Powerful fighter - Goku with everything
goku = Fighter("Goku")
print(f"🧑 {goku.name} created")
print(goku.learn_ki_blast())
print(goku.learn_flight())
print(goku.gain_transformation("Super Saiyan", 50))

print(f"\n{goku.name} in action:")
print(f"{goku.name}: {goku.attack()}")
print(f"{goku.name}: {goku.fly_around()}")
print(f"{goku.name}: {goku.power_up()}")
```

This can be nice because not every object needs to have all the abilities or components. This does not force you into maintaining uneeded attributes.

# PolyMorphing Power Rangers

This is like the Todd Howard of programming. No matter the object just call this function and it "just works". Polymorphism is a way for one method or one symbol to represent or perform different types. I just dont understand why this gets taught as if OOP made this unique. Here is a caveman example of this in python

```python
class Car:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Drive!")

class Boat:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Sail!")

class Plane:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Fly!")

car1 = Car("Ford", "Mustang")       #Create a Car object
boat1 = Boat("Ibiza", "Touring 20") #Create a Boat object
plane1 = Plane("Boeing", "747")     #Create a Plane object

for x in (car1, boat1, plane1):
  x.move()
```

Now the `move` will function for any of the class instances. OOP did not invent this or strengthen it. It just packaged it differently.

## Old Timers be Morphing

Get ready for this throwback. ALGOL 68 (1968) had polymorphism.

```
# Simple Union type example - ALGOL 68's form of polymorphism
BEGIN
  # Define union type for numbers
  MODE NUMBER = UNION(INT, REAL);
  
  # Polymorphic procedure using CASE to handle different types
  PROC print number = (NUMBER n) VOID:
    CASE n IN
      (INT i): print(("Integer: ", whole(i, 0))),
      (REAL r): print(("Real: ", fixed(r, 8, 2)))
    ESAC;
  
  # Create different number types
  NUMBER int num = 42;
  NUMBER real num = 3.14159;
  
  # Polymorphic usage - same procedure works with both types
  print number(int num);
  print(newline);
  print number(real num);
  print(newline)
END
```

This code is old as dirt and it handles both `INT` and `REAL`. This is essentially what OOP is doing so. In C you can get polymorphism via void pointers, VTables, etc...

Polymorphism is very powerful but OOP being some special harbinger is not true.

# Abstraction

I have a lot to say on this area but for this post we will keep it brief and expand upon it at a later date. Abstraction involves hiding details from the user and only exposing the necessary features of an object. This doesn't even make sense to be a pillar of OOP. This has been done since the dawn of programming. It's like saying food fed by a spoon is a pillar of humanity. Food is for sure but I don't know about the spoon.

So in the ideal world you get a nice clean object that you pray never crashes because if you have to untangle all the abstractions you are screwed my friend. You end up with all these strange hierarchical class structures that make no sense.

Sometimes you need to know or have access to the important parts of the program. If your database ORM library is having issues with optimized queries you need to be able to know why this is happening and how to correct it.

Regardless this just is not a pillar of OOP. If you want to try and claim it for OOP be my guest but it is a huge turnoff for me.

Here's an example of "hiding" a filter function

```python
def map_function(func, items):
    return [func(item) for item in items]
```

You only expose the nice function to the user. This is a bad example and a stupid example but I see this exact thing all the time. It would be better to directly use the code that does the real operations.

For now know that this is not unique to OOP at all.


# Conclusion: OOP Repackaged These Concepts

I will give credit where it's due. OOP packaged these concepts and made them easy to understand for everyone. OOP when used properly can be powerful. My focus for this article was to show that OOP being taught as the champion of these 4 pillars is not neccearily true and many other non OOP languages do these pillars better.

OOP is a tool and you need to learn when to not use certain things. Just because you can inherit you probably should think twice before doing it.