package snickrs.ailibrary.functions;

import java.util.Random;
import snickrs.ailibrary.env.Parameters;

public class Functions {
	public static Random random = new Random(Parameters.seed);
	public static double exp(double x) {
		return Math.exp(x);
	}
	public static double log(double base, double x) {
		if(base == Math.E) return Math.log(x);
		else if(base == 10.0d) return Math.log10(x);
		else return Math.log(x) / Math.log(base); // change of base formula
	}
	public static double sigmoid(double x) {
		return 1.0d / (1.0d + Math.exp(-x));
	}
	public static double hard_sigmoid(double x) {
		return Math.max(0.0d, Math.min(1.0d, x/6.0d + 0.5d));
	}
	public static double signum(double x) {
		return Math.signum(x);
	}
	public static double sin(double x) {
		return Math.sin(x);
	}
	public static double cos(double x) {
		return Math.cos(x);
	}
	public static double tan(double x) {
		return Math.tan(x);
	}
	public static double asin(double x) {
		return Math.asin(x);
	}
	public static double acos(double x) {
		return Math.acos(x);
	}
	public static double atan(double x) {
		return Math.atan(x);
	}
	public static double sinh(double x) {
		return Math.sinh(x);
	}
	public static double cosh(double x) {
		return Math.cosh(x);
	}
	public static double tanh(double x) {
		return Math.tanh(x);
	}
	public static double asinh(double x) {
		return Math.log(x + Math.sqrt(Math.pow(x, 2.0d) + 1.0d));
	}
	public static double acosh(double x) {
		return Math.log(x + Math.sqrt(Math.pow(x, 2.0d) - 1.0d));
	}
	public static double atanh(double x) {
		return 0.5d * Math.log((1.0d+x)/(1.0d-x));
	}
	public static double log(double x) {
		return Math.log(x);
	}
	public static double log10(double x) {
		return Math.log10(x);
	}
	public static double log1p(double x) {
		return Math.log1p(x);
	}
	public static double sqrt(double x) {
		return Math.sqrt(x);
	}
	public static double cbrt(double x) {
		return Math.cbrt(x);
	}
	public static double abs(double x) {
		return Math.abs(x);
	}
	public static double hard_tanh(double x) {
		return Math.max(-1.0d, Math.min(1.0d, x));
	}
	public static double rectified_tanh(double x) {
		return Math.max(0.0f, Math.tanh(x));
	}
	public static double relu(double x) {
		return Math.max(0.0d, x);
	}
	public static double relu6(double x) {
		return Math.max(0.0d, Math.min(x, 6.0d));
	}
	public static double leaky_relu(double alpha, double x) {
		return Math.max(alpha*x, x);
	}
	public static double rrelu(double lower, double upper, double x) {
		return Math.max(random.nextDouble(lower, upper)*x, x);
	}
	public static double elu(double alpha, double x) {
		if(x >= 0.0d) return x;
		else return alpha*(Math.exp(x) - 1.0d);
	}
	public static double silu(double alpha, double x) {
		return x * sigmoid(alpha*x);
	}
	public static double hard_silu(double x) {
		return Math.max(0.0d, Math.min(x, x*(x+3.0d)/6.0d));
	}
	public static double softplus(double x) {
		return Math.log(1.0d + Math.exp(x));
	}
	public static double mish(double x) {
		return x * Math.tanh(softplus(x));
	}
	public static double step(double x) {
		return x <= 0.0d ? 0.0d : 1.0d;
	}
	public static double identity(double x) {
		return x;
	}
	public static double pow(double x, double a) {
		return Math.pow(x, a);
	}
	public static double square(double x) {
		return Math.pow(x, 2.0d);
	}
	public static double cube(double x) {
		return Math.pow(x, 3.0d);
	}
	public static double selu(double x) {
		double alpha = 1.67326324d;
		double scale = 1.05070098d;
		return x > 0.0d ? scale * x : scale * alpha * (Math.exp(x) - 1.0d);
	}
    public static double erf(double x) {
        double a1 =  0.254829592d;
        double a2 = -0.284496736d;
        double a3 =  1.421413741d;
        double a4 = -1.453152027d;
        double a5 =  1.061405429d;
        double p  =  0.3275911d;
        double sign = Math.signum(x);
        x = Math.abs(x);
        double t = 1.0d/(1.0d + p*x);
        double y = 1.0d - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*Math.exp(-x*x);
        return sign*y;
    }
    public static double gamma(double x) {
		double[] p = {0.99999999999980993d, 676.5203681218851d, -1259.1392167224028d, 771.32342877765313d, -176.61502916214059d, 12.507343278686905d, -0.13857109526572012d, 9.9843695780195716e-6d, 1.5056327351493116e-7d};
		int g = 7;
		if(x < 0.5) return Math.PI / (Math.sin(Math.PI * x)*gamma(1-x));
		x -= 1;
		double a = p[0];
		double t = x+g+0.5;
		for(int i = 1; i < p.length; i++){
			a += p[i]/(x+i);
		}
		return Math.sqrt(2*Math.PI)*Math.pow(t, x+0.5)*Math.exp(-t)*a;
	}
    public static double digamma(double x) {
    	double GAMMA = 0.577215664901532860606512090082d;
        double C_LIMIT = 49;
        double S_LIMIT = 1e-5;
        double F_M1_12 = -1d / 12;
        double F_1_120 = 1d / 120;
        double F_M1_252 = -1d / 252;
        if (!Double.isFinite(x)) {
            return x;
        }
        double digamma = 0;
        if (x < 0) {
            digamma -= Math.PI / Math.tan(Math.PI * x);
            x = 1 - x;
        }
        if (x > 0 && x <= S_LIMIT) {
            return digamma - GAMMA - 1 / x;
        }
        while (x < C_LIMIT) {
            digamma -= 1 / x;
            x += 1;
        }
        final double inv = 1 / (x * x);
        digamma += Math.log(x) - 0.5 / x + inv * (F_M1_12 + inv * (F_1_120 + F_M1_252 * inv));
        return digamma;
    }
    // computes the principle k = 0 branch
    public static double lambertw(double z) {
        double EM1_FACT_0 = 0.57086272525975246; // 0x1.24481e7efdfcep-1
        double EM1_FACT_1 = 0.64442715366299452; // 0x1.49f25b1b461b7p-1
        double QE1 = 2.7182818284590452 * 0.25;  // 0x1.5bf0a8b145769p-1
        if (Double.isNaN(z) || Double.isInfinite(z) || z == 0.0) return z + z;
        if (Math.abs(z) < 1.9073486328125e-6) return fma(fma(1.5, z, -1.0) * z, z, z);

        double redz = fma(EM1_FACT_0, EM1_FACT_1, z); // z + exp(-1)
        double w, r, t, y, num, den, rden, e;

        if (redz < 0.01025390625) { // expansion at -(exp(-1))
            r = Math.sqrt(redz);
            w = -7.8466654751155138;  // -0x1.f62fc463917ffp+2
            w = fma(w, r, 10.0241581340373877); // 0x1.40c5e74773ef5p+3
            w = fma(w, r, -8.1029379749359691); // -0x1.034b44947bba0p+3
            w = fma(w, r, 5.8322883145113726); // 0x1.75443634ead5fp+2
            w = fma(w, r, -4.1738796362609882); // -0x1.0b20d80dcb9acp+2
            w = fma(w, r, 3.0668053943936471); // 0x1.888d14440efd0p+1
            w = fma(w, r, -2.3535499689514934); // -0x1.2d41201913016p+1
            w = fma(w, r, 1.9366310979331112); // 0x1.efc70e3e0a0eap+0
            w = fma(w, r, -1.8121878855270763); // -0x1.cfeb8b968bd2cp+0
            w = fma(w, r, 2.3316439815968506); // 0x1.2a734f5b6fd56p+1
            w = fma(w, r, -1.0); // -0x1.0000000000000p+0
            return w;
        }
        y = fma(2.0, Math.sqrt(fma(QE1, z, 0.25)), 1.0);
        y = Math.log(fma(1.14956131, y, -0.14956131) / fma(0.4549574, Math.log(y), 1.0));
        w = fma(2.036, y, -1.0);
        for (int i = 0; i < 3; i++) {
            t = w / (1.0 + w);
            w = fma(Math.log(z / w), t, t);
        }

        // Fine-tune approximation with a single Newton iteration
        e = Math.exp(w) * 0.125;
        num = fma(w, e, -0.125 * z);
        den = fma(w, e, e);
        rden = 1.0 / den;
        w = fma(-num, rden, w);
        return w;
    }
    
    public static double erfinv(double a) {
        double p, r, t;
        t = fma(a, -a, 1.0d);
        t = Math.log(t);

        if (Math.abs(t) > 6.125d) { // maximum ulp error = 2.35793
            p = 3.03697567e-10d; //  0x1.4deb44p-32 
            p = fma(p, t, 2.93243101e-8d); //  0x1.f7c9aep-26 
            p = fma(p, t, 1.22150334e-6d); //  0x1.47e512p-20 
            p = fma(p, t, 2.84108955e-5d); //  0x1.dca7dep-16 
            p = fma(p, t, 3.93552968e-4d); //  0x1.9cab92p-12 
            p = fma(p, t, 3.02698812e-3d); //  0x1.8cc0dep-9 
            p = fma(p, t, 4.83185798e-3d); //  0x1.3ca920p-8 
            p = fma(p, t, -2.64646143e-1d); // -0x1.0eff66p-2 
            p = fma(p, t, 8.40016484e-1d); //  0x1.ae16a4p-1 
        } else { // maximum ulp error = 2.35002
            p = 5.43877832e-9f; //  0x1.75c000p-28 
            p = fma(p, t, 1.43285448e-7d); //  0x1.33b402p-23 
            p = fma(p, t, 1.22774793e-6d); //  0x1.499232p-20 
            p = fma(p, t, 1.12963626e-7d); //  0x1.e52cd2p-24 
            p = fma(p, t, -5.61530760e-5d); // -0x1.d70bd0p-15 
            p = fma(p, t, -1.47697632e-4d); // -0x1.35be90p-13 
            p = fma(p, t, 2.31468678e-3d); //  0x1.2f6400p-9 
            p = fma(p, t, 1.15392581e-2d); //  0x1.7a1e50p-7 
            p = fma(p, t, -2.32015476e-1d); // -0x1.db2aeep-3 
            p = fma(p, t, 8.86226892e-1d); //  0x1.c5bf88p-1 
        }

        r = a * p;
        return r;
    }

    public static double erfi(double x) {
        return 2.0d/Math.sqrt(Math.PI)*integrate(t -> Math.exp(Math.pow(t, 2.0d)), 0.0d, x);
    }
    public static double dawson_f(double x) {
        return Math.exp(-Math.pow(x, 2.0d))*integrate(t -> Math.exp(Math.pow(t, 2.0d)), 0.0d, x);
    }

    // Helper method to compute fused multiply-add
    private static double fma(double a, double b, double c) {
        return a * b + c;
    }
    
    public static double zeta(double s, double a) {
        double sum = 0.0d;
        double EPSILON = 1e-16;
        int MAX_ITER = 100000; // slow convergence at times unless a crazy amount of terms
        for (int n = 0; n < MAX_ITER; n++) {
            double term = 1.0d / Math.pow(n+a, s);
            sum += term;
            if (term < EPSILON) {
                break;
            }
        }
        return sum;
    }
    public static double eta(double s) {
        double sum = 0.0d;
        double EPSILON = 1e-16;
        int MAX_ITER = 100000; // slow convergence at times unless a crazy amount of terms
        for (int n = 1; n < MAX_ITER; n++) {
            double term = Math.pow(-1.0d, n-1.0d) / Math.pow(n, s);
            sum += term;
            if (term < EPSILON) {
                break;
            }
        }
        return sum;
    }
    // gamma(p, x)
    public static double multigamma(double p, double x) {
        double product = Math.pow(Math.PI, p * (p - 1.0d) / 4.0d);
        for (int j = 1; j <= p; j++) {
            product *= gamma(x + (1.0d-j)/2.0d);
        }
        return product;
    }
    
    // only for integer values, see gamma function for non-integers
    public static double factorial(double n) {
        double result = 1.0d;
        for (int i = 1; i <= (int) n; i++) {
            result *= i;
        }
        return result;
    }
    public static double pochhammer(double x, double n) {
        return gamma(x+n)/gamma(x);
    }
    public static double choose(int n, int k) {
    	if (k < 0 || k > n) return 0.0d;
        if (k == 0 || k == n) return 1.0d;
        double result = 1.0d;
        for (int i = 1; i <= k; i++) {
            result *= (n - k + i) / i;
        }
        return result;
    }
    public static double choose(double n, double k) {
        if (k == 0 || k == n) return 1.0d;
        return 1.0d/(k*beta(k, n-k+1.0d));
    }
    // m-th derivative of the digamma function, z is the input
    public static double polygamma(double m, double z) {
    	double sum = 0.0d;
    	int MAX_ITER = 100000;
    	double EPSILON = 1e-16;
    	for(int k = 0; k <= MAX_ITER; k++) {
    		double term = 1.0d/Math.pow(z+k, m+1);
    		sum += term;
    		if(term < EPSILON) {
    			break;
    		}
    	}
    	return sum * Math.pow(-1.0d, m+1.0d) * factorial(m);
    }
    
    public static double polylog(double n, double x) {
        // polylogarithm: Li(n, x) = sum(x^i/i^n, 1, inf);
    	double EPSILON = 1e-16d;
        double sum = 0.0d;
        double term;
        double k = 1.0d;
        do {
            term = Math.pow(x, k) / Math.pow(k, n);
            sum += term;
            k++;
        } while (Math.abs(term) > EPSILON);
        return sum;
    }
    public static double ci_helper(double t) {
        // Ci(x) = C + ln(x) + int((cos(t)-1)/t, 0, x);
    	if(t == 0.0d) return 0.0d;
    	return (Math.cos(t)-1.0d)/t;
    }
    public static double atanint_helper(double t) {
        // Atanint(x) = int(atan(t)/t, 0, x);
    	if(t == 0.0d) return 1.0d;
    	return (Math.atan(t))/t;
    }
    public static double si_helper(double t) {
    	// Si(x) = int(sin(t)/t, 0, x);
    	if(t == 0.0d) return 1.0d;
    	return Math.sin(t)/t;
   }
    public static double chi_helper(double t) {
        // Chi(x) = C + ln(x) + int((cosh(t)-1)/t, 0, x);
    	if(t == 0.0d) return 0.0d;
    	return (Math.cosh(t)-1.0d)/t;
    }
    public static double shi_helper(double t) {
    	// Shi(x) = int(sinh(t)/t, 0, x);
    	if(t == 0.0d) return 1.0d;
    	return Math.sinh(t)/t;
   }
    public static double ei_helper(double t) {
        // Ei(x) = -int(e^(-t)/t, -x, inf);
    	// E1(x) = int(e^(-t)/t, x, inf);
    	if(t == 0.0d) return 0.0d;
    	return Math.exp(-t)/t;
    }
    public static double ei_helper(double n, double x, double t) {
        // E_n(x) = int(e^(-xt)/(t^n), 1, inf);
    	if(t == 0.0d) return 0.0d;
    	return Math.exp(-x*t)/Math.pow(t, n);
    }
    public static double debye_helper(double n, double t) {
        // D(n, x)
    	if(t == 0.0d) return n != 0 ? 0.0d : 1.0d;
    	return Math.pow(t, n)/Math.expm1(t);
    }
    public static double transport_j_helper(double n, double t) {
        // D(n, x)
    	if(t == 0.0d) return n != 2 ? 0.0d : 1.0d;
    	return Math.pow(t, n)*Math.exp(t)/Math.pow(Math.expm1(t), 2.0d);
    }
    public static double fermi_dirac_helper(double j, double x, double t) {
    	return Math.pow(t, j)/(Math.exp(t-x)+1.0d);
    }
    public static double li_helper(double t) {
    	//li(x)
    	if(t == 1.0d) return 0.0d;
    	return 1.0d/Math.log(t);
    }
    public static double fresnel_c_helper(double t) {
    	// C(x)
    	return Math.cos(0.5d*Math.PI*Math.pow(t, 2.0d));
    }
    public static double fresnel_s_helper(double t) {
    	// S(x)
    	return Math.sin(0.5d*Math.PI*Math.pow(t, 2.0d));
    }
    public static double airy_ai_helper(double x, double t) {
    	// Ai(x)
    	return Math.cos(Math.pow(t, 3.0d)/3.0d + x*t);
    }
    public static double airy_bi_helper(double x, double t) {
    	// Bi(x)
    	return Math.exp(-Math.pow(t, 3.0d)/3.0d + x*t) + Math.sin(Math.pow(t, 3.0d)/3.0d + x*t);
    }
    public static double airy_ai_prime_helper(double x, double t) {
    	// Ai'(x)
    	return t*Math.sin(Math.pow(t, 3.0d)/3.0d + x*t);
    }
    public static double airy_bi_prime_helper(double x, double t) {
    	// Bi'(x)
    	return t*(Math.exp(-Math.pow(t, 3.0d)/3.0d + x*t) + Math.cos(Math.pow(t, 3.0d)/3.0d + x*t));
    }
    public static double inc_gamma_helper(double a, double t) {
    	// gamma(a, x)
    	double h = Math.pow(t, a-1.0d)*Math.exp(-t);
    	if(Double.isNaN(h)) return 0.0d;
    	return h;
    }
    public static double inc_beta_helper(double a, double b, double t) {
    	// beta(x, a, b)
    	double h = Math.pow(t, a-1.0d)*Math.pow(1.0d-t, b-1.0d);
    	if(Double.isNaN(h)) return 0.0d;
    	return h;
    }
    public static double elliptic_e_helper(double m, double t) {
    	// E(x | m), m = k^2
    	double x = Math.sqrt(1.0d-m*Math.pow(Math.sin(t), 2.0d));
    	if(Double.isNaN(x)) return 0.0d;
    	return x;
    }
    public static double elliptic_f_helper(double m, double t) {
    	// F(x | m), m = k^2
    	double x = 1.0d/Math.sqrt(1.0d-m*Math.pow(Math.sin(t), 2.0d));
    	if(Double.isNaN(x)) return 0.0d;
    	return x;
    }
    public static double elliptic_pi_helper(double n, double m, double t) {
    	// PI(n; x | m)
    	double x = 1.0d/((1.0d-n*Math.pow(Math.sin(t), 2.0d))*Math.sqrt(1.0d-m*Math.pow(Math.sin(t), 2.0d)));
    	if(Double.isNaN(x)) return 0.0d;
    	return x;
    }
    // numerical integration by adaptive quadrature
    public static double integrate(Func f, double a, double b) {
    	double EPSILON = 1e-16d;
    	double h = b - a;
    	double c = (a + b) / 2.0d;
    	double d = (a + c) / 2.0d;
    	double e = (b + c) / 2.0d;
    	double fa = f.apply(a);
    	double fb = f.apply(b);
    	double fc = f.apply(c);
    	double fd = f.apply(d);
    	double fe = f.apply(e);
        double Q1 = h/6.0d  * (fa + 4.0d*fc + fb);
        double Q2 = h/12.0d * (fa + 4.0d*fd + 2.0d*fc + 4.0d*fe + fb);
        if (Math.abs(Q2 - Q1) <= EPSILON) {
        	return Q2 + (Q2 - Q1) / 15.0d;
        } else {
        	return integrate(f, a, c) + integrate(f, c, b);
        }
    }
    public static double cos_integral(double x) {
		return 0.577215664901532861d + Math.log(x) + integrate(f -> ci_helper(f), 0.0d, x);
    }
    public static double sin_integral(double x) {
		return integrate(f -> si_helper(f), 0.0d, x);
    }
    public static double transport_j_integral(double n, double x) {
		return integrate(f -> transport_j_helper(n, f), 0.0d, x);
    }
    public static double fermi_dirac_integral(double j, double x) {
    	double EPSILON = 100.0d;
		return integrate(f -> fermi_dirac_helper(j, x, f), 0.0d, EPSILON)/gamma(j+1.0d);
    }
    public static double atan_integral(double x) {
		return integrate(f -> atanint_helper(f), 0.0d, x);
    }
    public static double cosh_integral(double x) {
		return 0.577215664901532861d + Math.log(x) + integrate(f -> chi_helper(f), 0.0d, x);
    }
    public static double sinh_integral(double x) {
		return integrate(f -> shi_helper(f), 0.0d, x);
    }
    public static double log_integral(double x) {
    	double EPSILON = 1e-16d; // you can set this as low as you want, Double.MIN_VALUE is the lowest
    	if(x < 1.0d) {
    		return integrate(f -> li_helper(f), 0.0d, x);
    	} else if (x < 1.45136923488338105d) {
    		return integrate(f -> li_helper(f), 0.0d, 1.0d-EPSILON) + integrate(f -> li_helper(f), 1.0d+EPSILON, x);
    	}
    	return integrate(f -> li_helper(f), 1.45136923488338105d, x);
    }
    public static double exp_integral(double x) {
    	double EPSILON = 1e-16d; // again can be set as low as desired
    	return -integrate(f -> ei_helper(f), -x, 0.0d-EPSILON) - integrate(f -> ei_helper(f), 0.0d+EPSILON, Double.MAX_VALUE);
    }
    public static double exp_integral(double n, double x) {
    	return integrate(f -> ei_helper(n, x, f), 1.0d, Double.MAX_VALUE);
    }
    public static double e1_integral(double x) {
        return integrate(f -> ei_helper(f), x, Double.MAX_VALUE);
    }
    public static double debye1_integral(double n, double x) {
        return integrate(f -> debye_helper(n, f), 0.0d, x);
    }
    public static double debye2_integral(double n, double x) {
    	double EPSILON = 100.0d; // cutoff value for integration;
        return integrate(f -> debye_helper(n, f), x, EPSILON);
    }
    public static double fresnel_c_integral(double x) {
        return integrate(f -> fresnel_c_helper(f), 0.0d, x);
    }
    public static double fresnel_s_integral(double x) {
        return integrate(f -> fresnel_s_helper(f), 0.0d, x);
    }
    public static double uppergamma(double a, double x) {
    	double EPSILON = 100; // gets hard to integrate at anything above this
        return integrate(f -> inc_gamma_helper(a, f), x, EPSILON);
    }
    public static double lowergamma(double a, double x) {
        return integrate(f -> inc_gamma_helper(a, f), 0.0d, x);
    }
    public static double incomplete_beta(double x, double a, double b) {
        return integrate(f -> inc_beta_helper(a, b, f), 0.0d, x);
    }
    public static double airy_ai_integral(double x) {
    	double EPSILON = 10; // gets too slow to integrate at anything above this
        return integrate(f -> airy_ai_helper(x, f), 0.0d, EPSILON)/Math.PI;
    }
    public static double airy_bi_integral(double x) {
    	double EPSILON = 10; // gets too slow to integrate at anything above this
        return integrate(f -> airy_bi_helper(x, f), 0.0d, EPSILON)/Math.PI;
    }
    public static double airy_ai_prime_integral(double x) {
    	double EPSILON = 10; // gets too slow to integrate at anything above this
        return -integrate(f -> airy_ai_prime_helper(x, f), 0.0d, EPSILON)/Math.PI;
    }
    public static double airy_bi_prime_integral(double x) {
    	double EPSILON = 10; // gets too slow to integrate at anything above this
        return integrate(f -> airy_bi_prime_helper(x, f), 0.0d, EPSILON)/Math.PI;
    }
    public static double elliptic_f_integral(double x, double m) {
    	// F(x | m), m = k^2
        return integrate(f -> elliptic_f_helper(m, f), 0.0d, x);
    }
    public static double elliptic_e_integral(double x, double m) {
    	// E(x | m), m = k^2
        return integrate(f -> elliptic_e_helper(m, f), 0.0d, x);
    }
    public static double complete_elliptic_k_integral(double m) {
    	// K(m), m = k^2
        return integrate(f -> elliptic_f_helper(m, f), 0.0d, Math.PI/2.0d);
    }
    public static double complete_elliptic_e_integral(double m) {
    	// E(m), m = k^2
        return integrate(f -> elliptic_e_helper(m, f), 0.0d, Math.PI/2.0d);
    }
    public static double elliptic_pi_integrator(double n, double m, double a, double b) {
    	double EPSILON = 1e-16d;
    	double h = b - a;
    	double c = (a + b) / 2.0d;
    	double d = (a + c) / 2.0d;
    	double e = (b + c) / 2.0d;
    	double fa = elliptic_pi_helper(n, m, a);
    	double fb = elliptic_pi_helper(n, m, b);
    	double fc = elliptic_pi_helper(n, m, c);
    	double fd = elliptic_pi_helper(n, m, d);
    	double fe = elliptic_pi_helper(n, m, e);
        double Q1 = h/6.0d  * (fa + 4.0d*fc + fb);
        double Q2 = h/12.0d * (fa + 4.0d*fd + 2.0d*fc + 4.0d*fe + fb);
        if (Math.abs(Q2 - Q1) <= EPSILON) {
        	return Q2 + (Q2 - Q1) / 15.0d;
        } else {
        	return elliptic_pi_integrator(n, m, a, c) + elliptic_pi_integrator(n, m, c, b);
        }
    }
    public static double elliptic_pi_integral(double n, double x, double m) {
    	// PI(n; x | m), m = k^2
        return elliptic_pi_integrator(n, m, 0.0d, x);
    }
    public static double complete_elliptic_pi_integral(double n, double m) {
    	// PI(n | m), m = k^2
        return elliptic_pi_integrator(n, m, 0.0d, Math.PI/2);
    }
    public static double beta(double x, double y) {
    	return gamma(x)*gamma(y)/gamma(x+y);
    }
    
	public static double p0(double x) {
		return 0.5d * (1 + erf(x / Math.sqrt(2)));
	}
	public static double gelu(double x, boolean approximate) {
		if(approximate) return 0.5d*x*(1.0d+Math.tanh(Math.sqrt(2.0d/Math.PI)*(x+0.044715d*Math.pow(x, 3.0d))));
		else return x * p0(x);
	}
	public static double softsign(double x) {
		return x / (1.0d + Math.abs(x));
	}
	public static double celu(double alpha, double x) {
		if(x >= 0.0d) return x;
		else return alpha*(Math.exp(x/alpha)-1.0d);
	}
	public static double softshrink(double alpha, double x) {
		if(x > alpha) return x-alpha;
		else if(x < -alpha) return x+alpha;
		else return 0.0d;
	}
	public static double hard_shrink(double alpha, double x) {
		return (x >= -alpha && x <= alpha) ? 0.0d : x;
	}
}
@FunctionalInterface
interface Func {
	public double apply(double x);
}
