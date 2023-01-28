from unittest import TestCase

import numpy as np


class TestEx01(TestCase):

    def runTest(self):
        import docs.examples.ex01 as ex01
        self.assertAlmostEqual(np.max(ex01.x), 0.9988684986378847)


class TestEx02(TestCase):

    def runTest(self):
        import docs.examples.ex02 as ex02
        self.assertAlmostEqual(np.max(ex02.x), 2.47051146667486)


class TestEx03(TestCase):

    def runTest(self):
        import docs.examples.ex03 as ex03
        self.assertAlmostEqual(np.max(ex03.x.real), 4e-07)
        self.assertAlmostEqual(np.max(ex03.x.imag), 5.647644273654593e-10)


class TestEx04(TestCase):

    def runTest(self):
        import docs.examples.ex04 as ex04
        self.assertAlmostEqual(np.max(ex04.x), 7.523543025631155e-06)
        self.assertAlmostEqual(np.min(ex04.x), -0.12461559817005023)


class TestEx05(TestCase):

    def runTest(self):
        import docs.examples.ex05 as ex05
        x = np.max(ex05.x.reshape(-1, 4), axis=0)
        self.assertAlmostEqual(x[0], 1.00029798e+00)
        self.assertAlmostEqual(x[1], 5.93550842e+00)
        self.assertAlmostEqual(x[2], 7.52354303e-06)
        self.assertAlmostEqual(x[3], 0.00000000e+00)

        x = np.min(ex05.x.reshape(-1, 4), axis=0)
        self.assertAlmostEqual(x[0], -3.93550842)
        self.assertAlmostEqual(x[1], 0)
        self.assertAlmostEqual(x[2], -0.1246156)
        self.assertAlmostEqual(x[3], -0.01)


class TestEx06(TestCase):

    def runTest(self):
        import docs.examples.ex06 as ex06
        x = np.max(np.abs(ex06.x.reshape(-1, 4)), axis=0)
        self.assertAlmostEqual(x[0], 5.887689e-06)
        self.assertAlmostEqual(x[1], 5.887689e-06)
        self.assertAlmostEqual(x[2], 4.000000e-07)
        self.assertAlmostEqual(x[3], 0)


class TestEx07(TestCase):

    def runTest(self):
        import docs.examples.ex07 as ex07
        self.assertAlmostEqual(np.max(ex07.x), 0.00039345332035954796)
        self.assertAlmostEqual(np.min(ex07.x), -0.029510886209319683)


class TestEx08(TestCase):

    def runTest(self):
        import docs.examples.ex08 as ex08
        x = np.max(ex08.x.reshape(-1, 4), axis=0)
        self.assertAlmostEqual(x[0], 1.01558306e+00)
        self.assertAlmostEqual(x[1], 2.16880414e+00)
        self.assertAlmostEqual(x[2], 3.93453320e-04)
        self.assertAlmostEqual(x[3], 0)

        x = np.min(ex08.x.reshape(-1, 4), axis=0)
        self.assertAlmostEqual(x[0], -0.16880414)
        self.assertAlmostEqual(x[1], 0)
        self.assertAlmostEqual(x[2], -0.02951089)
        self.assertAlmostEqual(x[3], -0.002)


class TestEx09(TestCase):

    def runTest(self):
        import docs.examples.ex09 as ex09
        x = np.max(np.abs(ex09.x.reshape(-1, 4)), axis=0)
        self.assertAlmostEqual(x[0], 3.99566449e-03)
        self.assertAlmostEqual(x[1], 3.25574710e-02)
        self.assertAlmostEqual(x[2], 5.00000000e-03)
        self.assertAlmostEqual(x[3], 2.46398240e-05)


class TestEx12(TestCase):

    def runTest(self):
        import docs.examples.ex12 as ex12
        x = np.max(ex12.x[ex12.mesh.y <= -ex12.capsol.d])
        self.assertAlmostEqual(x, 0.5461501998814416)
