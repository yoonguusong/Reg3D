class Person():
    def __init__(self, name):
        self.name = name

class EmailPerson(Person):
    def __init__(self, name, email):
        super().__init__(name)
        self.email = email

hunter = Person("Song")
print(hunter.name)

bob = EmailPerson('bob', 'syg@naver')
print(bob.name)
print(bob.email)

class Car():
    def exclaim(self):
        print("im a car")


class Yugo(Car):
    def exclaim(self):
        print("im a different car")

    def need_a_push(self):
        print("a little help")


give1 = Car()
give2 = Yugo()
give1.exclaim()
give2.exclaim()
give2.need_a_push()


class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

class Square(Rectangle):
    def __init__(self, length):
        super().__init__(length, length)

a = Square(4)
print(a.area())

class Duck():
    '''
    hello
    '''
    def __init__(self, input_name):
        self.hidden_name = input_name
        @property
        def name(self):
            print('inside the getter')
            return self.hidden_name

        @name.setter
        def name(self, input_name):
            print('inside the setter')
            self.hidden_name = input_name
        name = property(get_name, set_name)


def document_it(func):
    def new_function(*args, **kwargs):
        print('running function :', func.__name__)
        print('Positional arguments : ', args)
        print('keyword arguments : ', kwargs)
        result = func(*args, **kwargs)
        print('result : ', result)
        return result
    return new_function

def square_it(func):
    def new_function(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * result
    return new_function

@document_it
@square_it
def add_ints(a,b):
    return a+b


@square_it
@document_it
def add_ints(a,b):
    return a+b

add_ints(3,5)


print('add_ints(',3,5,') : ', add_ints(3,5))

cooler_add_ints = document_it(add_ints)
cooler_add_ints(3,5)

animal = 'a'
def print_global():
    animal = 'b'
    print('local : ', animal)
print(animal)
print_global()
print(animal)


class Duck():
    '''
    hello
    '''
    def __init__(self, input_name):
        self.__name = input_name
        @property
        def name(self):
            print('inside the getter')
            return self.hidden_name

        @name.setter
        def name(self, input_name):
            print('inside the setter')
            self.__name = input_name


name1= Duck('name1')
name1._Duck__name