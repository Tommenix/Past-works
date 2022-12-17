import unittest

from fibonacci import *
from egyptian import *

# ground truth fibonacci numbers
fibs = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

class TestFibonacci(unittest.TestCase):

	def setUp(self):
		pass

	def test_recursive(self):
		for n, ans in enumerate(fibs):
			self.assertEqual(fibonacci_recursive(n), ans)

	def test_iter(self):
		for n, ans in enumerate(fibs):
			self.assertEqual(fibonacci_iter(n), ans)

	def test_power(self):
		for n, ans in enumerate(fibs):
			self.assertEqual(fibonacci_power(n), ans)


class TestEgyptian(unittest.TestCase):

	def setUp(self):
		pass

	def test_mult(self):
		for a in range(30):
			for n in range(30):
				self.assertEqual(egyptian_multiplication(a, n), a * n)


	def test_pow(self):
		for a in range(30):
			for n in range(3):
				self.assertEqual(power(a, n), a ** n)

		for a in range(3):
			for n in range(10):
				self.assertEqual(power(a, n), a ** n)
