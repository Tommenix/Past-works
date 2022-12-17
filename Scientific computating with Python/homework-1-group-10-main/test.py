import unittest

from functions import *

tol = 1e-8

"""
Tests operate by checking that functions evaluate to the correct answer
"""

class Test0B(unittest.TestCase):

	def setUp(self):
		pass

	def test_constant(self):
		for c in [-1.0, 0.0, 3.0]:
			cx = Constant(c)
			self.assertTrue(isinstance(cx, AbstractFunction))
			self.assertTrue(isinstance(cx, Polynomial))
			x = np.linspace(-1,1,100)
			self.assertTrue(np.all(cx(x) - c < tol))

	def test_scale(self):
		for c in [-1.0, 1.0, 3.0]:
			cx = Scale(c)
			self.assertTrue(isinstance(cx, AbstractFunction))
			self.assertTrue(isinstance(cx, Polynomial))
			x = np.linspace(-1,1,100)
			self.assertTrue(np.all(cx(x) - c*x < tol))


class Test0C(unittest.TestCase):

	def setUp(self):
		pass

	def test_compose(self):
		f = Compose(Polynomial(1,0,0,0), Polynomial(1,0,0))
		self.assertTrue(isinstance(f, AbstractFunction))
		x = np.linspace(-1,1,100)
		self.assertTrue(np.all(f(x) - x**6 < tol))

		fp = f.derivative()
		self.assertTrue(np.all(fp(x) - 6*x**5 < tol))


	def test_product(self):
		f = Product(Polynomial(1,0), Polynomial(1,0))
		self.assertTrue(isinstance(f, AbstractFunction))
		x = np.linspace(-1,1,100)
		self.assertTrue(np.all(f(x) - x**2 < tol))

		fp = f.derivative()
		self.assertTrue(np.all(fp(x) - 2*x < tol))


	def test_sum(self):
		f = Sum(Polynomial(1,0), Polynomial(1,1,0))
		self.assertTrue(isinstance(f, AbstractFunction))
		x = np.linspace(-1,1,100)
		self.assertTrue(np.all(f(x) - (x**2 + 2*x) < tol))

		fp = f.derivative()
		self.assertTrue(np.all(fp(x) - (2*x + 2) < tol))



class Test0D(unittest.TestCase):

	def setUp(self):
		pass

	def test_power(self):
		for n in [-1, -2, 1.5]:
			print(n)
			f = Power(n)
			self.assertTrue(isinstance(f, AbstractFunction))
			x = np.logspace(-2,2,100)
			self.assertTrue(np.all(f(x) - x**n < tol))

			fp = f.derivative()
			self.assertTrue(np.all(fp(x) - n*x**(n-1) < tol))


	def test_log(self):
		f = Log()
		self.assertTrue(isinstance(f, AbstractFunction))
		x = np.logspace(-2,5,100)
		self.assertTrue(np.all(f(x) - np.log(x) < tol))

		fp = f.derivative()
		self.assertTrue(np.all(fp(x) - x**-1 < tol))


	def test_exp(self):
		f = Exponential()
		self.assertTrue(isinstance(f, AbstractFunction))
		x = np.linspace(-2,5,100)
		self.assertTrue(np.all(f(x) - np.exp(x) < tol))

		fp = f.derivative()
		self.assertTrue(np.all(fp(x) - np.exp(x) < tol))


	def test_sin(self):
		f = Sin()
		self.assertTrue(isinstance(f, AbstractFunction))
		x = np.linspace(-2,5,100)
		self.assertTrue(np.all(f(x) - np.sin(x) < tol))

		fp = f.derivative()
		self.assertTrue(np.all(fp(x) - np.cos(x) < tol))


	def test_cos(self):
		f = Cos()
		self.assertTrue(isinstance(f, AbstractFunction))
		x = np.linspace(-2,5,100)
		self.assertTrue(np.all(f(x) - np.cos(x) < tol))

		fp = f.derivative()
		self.assertTrue(np.all(fp(x) - (-np.sin(x)) < tol))


class Test0E(unittest.TestCase):

	def setUp(self):
		pass

	def test_symbol(self):
		f = Symbolic('f')
		self.assertTrue(isinstance(f, AbstractFunction))
		self.assertEqual(str(f), "f({0})")
		self.assertEqual(f('x'), "f(x)")
		self.assertEqual(f(1), "f(1)")

		fp = f.derivative()
		self.assertEqual(str(fp), "f'({0})")

		g = Symbolic('g')
		h = f(g)
		self.assertEqual(str(h), "f(g({0}))")


class Test1A(unittest.TestCase):

	def setUp(self):
		pass

	def test_newton(self):
		f = Polynomial(1,1)
		x0 = 1.0
		x = newton_root(f, x0)
		self.assertAlmostEqual(x, -1.0)

		f = Polynomial(1,0,-2)
		x0 = 1.0
		x = newton_root(f, x0)
		self.assertAlmostEqual(x, np.sqrt(2))


class Test1B(unittest.TestCase):

	def setUp(self):
		pass


	def test_newton_extremum(self):

		f = Polynomial(1,0,-2)
		x0 = 1.0
		x = newton_extremum(f, x0)
		self.assertAlmostEqual(x, 0)


class Test2A(unittest.TestCase):

	def setUp(self):
		pass


	def test_polynomial(self):

		for x0 in [0, 1, 2]:
			f = Polynomial(1,2,3)
			Tf = f.taylor_series(x0, deg=3)
			self.assertTrue(isinstance(f, AbstractFunction))

			x = np.linspace(-1,1,100)
			self.assertTrue(np.all(f(x) - Tf(x) < tol))

			self.assertTrue(np.all(f.derivative()(x) - Tf.derivative()(x) < tol))


	def test_exp(self):

		f = Exponential()
		# degree 1 series
		Tf = f.taylor_series(0, deg=1)
		self.assertTrue(isinstance(f, AbstractFunction))

		x = np.linspace(-1,1,100)
		self.assertTrue(np.all(Tf(x) - (x + 1) < tol))

		# degree 2 series
		Tf = f.taylor_series(0, deg=2)
		self.assertTrue(isinstance(f, AbstractFunction))

		x = np.linspace(-1,1,100)
		self.assertTrue(np.all(Tf(x) - (0.5*x**2 + x + 1) < tol))
