package snickrs.ailibrary.functions;

import snickrs.ailibrary.env.Parameters;

public class Derivatives {
	public static double sigmoid(double x) {
		return Functions.sigmoid(x) * (1.0d - Functions.sigmoid(x));
	}
	public static double exp(double x) {
		return Math.exp(x);
	}
	public static double signum(double x) {
		return 0.0d;
	}
	public static double sin(double x) {
		return Math.cos(x);
	}
	public static double cos(double x) {
		return -Math.sin(x);
	}
	public static double tan(double x) {
		return 1.0d / Math.pow(Math.cos(x), 2.0d);
	}
	public static double asin(double x) {
		return 1.0d/Math.sqrt(1-Math.pow(x, 2.0d));
	}
	public static double acos(double x) {
		return -1.0d/Math.sqrt(1-Math.pow(x, 2.0d));
	}
	public static double sinh(double x) {
		return Math.cosh(x);
	}
	public static double log10(double x) {
		return 1.0d/(x*Math.log(10.0d));
	}
	public static double log1p(double x) {
		return 1.0d/(x+1.0d);
	}
	public static double cbrt(double x) {
		return 1.0d/(3.0d*Math.pow(x, 2.0d/3.0d));
	}
	public static double pow(double x, double a) {
		return a * Math.pow(x, a-1.0d);
	}
	public static double hard_sigmoid(double x) {
		if(x <= -3.0d) {
			return 0.0d;
		} else if (x < 3.0d) {
			return 1.0d/6.0d;
		}
		return 0.0d;
	}
	public static double tanh(double x) {
		return 1.0d - Math.pow(Math.tanh(x), 2.0d);
	}
	public static double cosh(double x) {
		return Math.sinh(x);
	}
	public static double abs(double x) {
		return Math.signum(x);
	}
	public static double log(double x) {
		return 1.0d/x;
	}
	public static double sqrt(double x) {
		return 0.5d*Math.pow(x, -0.5d);
	}
	public static double hard_tanh(double x) {
		if(x <= -1.0d) {
			return 0.0d;
		} else if (x < 1.0d) {
			return 1.0d;
		}
		return 0.0d;
	}
	public static double relu(double x) {
		return x <= 0.0d ? 0.0d : 1.0d;
	}
	public static double leaky_relu(double alpha, double x) {
		return x <= 0.0d ? -alpha : 1.0d;
	}
	public static double elu(double alpha, double x) {
		if(x >= 0.0d) return 1.0d;
		else return alpha*(Math.exp(x));
	}
	public static double silu(double alpha, double x) {
		double s = Functions.silu(alpha, x);
		return s + Functions.sigmoid(x) * (1.0d - s);
	}
	public static double hardsilu(double x) {
		if(x < -3.0d) {
			return 0.0d;
		} else if (x <= 3.0d) {
			return x / 3.0d + 0.5d;
		}
		return 1.0d;
	}
	public static double softplus(double x) {
		return Functions.sigmoid(x);
	}
	public static double mish(double x) {
		return Math.exp(x)*(Math.exp(3.0d*x)+4.0d*Math.exp(2.0d*x)+Math.exp(x)*(6.0d+4.0d*x)+4.0d*x+4.0d)/Math.pow(Math.pow(Math.exp(x)+1.0d,2.0d)+1.0d,2.0d);
	}
	public static double step(double x) {
		return 0.0d;
	}
	public static double identity(double x) {
		return 1.0d;
	}
	public static double square(double x) {
		return 2.0d * x;
	}
	public static double cube(double x) {
		return 3.0d * Math.pow(x, 2.0d);
	}
	public static double atan(double x) {
		return 1.0d / (Math.pow(x, 2.0d) + 1.0d);
	}
	public static double erf(double x) {
		return 2.0d*Math.exp(-Math.pow(x, 2.0d))/Math.sqrt(Math.PI);
	}
	public static double selu(double x) {
		double alpha = 1.67326324d;
		double scale = 1.05070098d;
		return x > 0.0d ? scale : scale * alpha * Math.exp(x);
	}
	public static double gelu(double x) {
		return Functions.p0(x) + x * Math.exp(-Math.pow(x, 2.0d)/2.0d) / Math.sqrt(2.0d * Math.PI);
	}
	public static double softsign(double x) {
		return 1.0d/Math.pow(Math.abs(x) + 1.0d, 2.0d);
	}
}
