package snickrs.ailibrary.autodiff.optimization;
@FunctionalInterface
public interface LearningRateScheduler {
	public double apply(int epoch, double lr);
}
