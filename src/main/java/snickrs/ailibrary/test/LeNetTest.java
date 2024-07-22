package snickrs.ailibrary;

import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.autodiff.activation.*;
import snickrs.ailibrary.autodiff.cnn.*;
import snickrs.ailibrary.autodiff.constraints.*;
import snickrs.ailibrary.autodiff.dropout.*;
import snickrs.ailibrary.autodiff.loss.*;
import snickrs.ailibrary.autodiff.math.*;
import snickrs.ailibrary.autodiff.merge.*;
import snickrs.ailibrary.autodiff.pooling.*;
import snickrs.ailibrary.autodiff.nn.*;
import snickrs.ailibrary.autodiff.normalization.*;
import snickrs.ailibrary.autodiff.optimization.*;
import snickrs.ailibrary.autodiff.regularization.*;
import snickrs.ailibrary.autodiff.reshape.*;
import snickrs.ailibrary.autodiff.weights.*;

import snickrs.ailibrary.datasets.*;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.enums.WeightsFormat;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import org.threeten.bp.LocalDateTime;
import org.threeten.bp.format.DateTimeFormatter;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class LeNetTest {
	public static DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm:ss.SSS");
	public static int training_steps = 0;
	public static int num_epochs = 1;
	public static int batch_size = 64; // bout 937 training steps to take
	public static int seed = 123; // random seed for reproducibility
	public static BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
	public static void main(String[] args) throws FileNotFoundException, IOException {
		Nd4j.getRandom().setSeed(seed); // random seed for reproducibility
		Nd4j.getMemoryManager().setAutoGcWindow(10000); // garbage collection every 10 seconds if all else fails
		println("begin mnist dataset test session");
		Thread t = new Thread(() -> {
			while (!Thread.interrupted()) {
				try {
					cmd();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		});
		t.start();
		println("initializing mnist dataset");
		Mnist dataset = new Mnist(batch_size, false, true);
		println("finished loading mnist dataset");
		println("initializing neural network");
		Placeholder input = new Placeholder("input", dataset.input_dims);
		Placeholder target = new Placeholder("target", dataset.label_dims);
		// create the LeNet model
		// layer 1 (convolutional)
		Variable w0 = new Variable("w0", new GlorotUniform(), 5, 5, 1, 20);
		Node c0 = new Conv2DNode(input, w0);
		Variable b0 = new Variable("b0", new Zeros(), c0.shape());
		Node a0 = new AdditionNode(c0, b0);
		// layer 2 (pooling)
		Node p0 = new MaxPooling2DNode(a0);
		// layer 3 (convolutional)
		Variable w1 = new Variable("w1", new GlorotUniform(), 5, 5, 20, 50);
		Node c1 = new Conv2DNode(p0, w1);
		Variable b1 = new Variable("b1", new Zeros(), c1.shape());
		Node a1 = new AdditionNode(c1, b1);
		// layer 4 (pooling)
		Node p1 = new MaxPooling2DNode(a1);
		// layer 5 (dense)
		Node flat = new FlattenNode(p1);
		Variable dw0 = new Variable("dw0", new HeNormal(), flat.shape()[1], 500);
		Variable db0 = new Variable("dw0", new Zeros(), 1, 500);
		Node d0 = new AdditionNode(new MmulNode(flat, dw0), db0);
		// layer 6 (dense)
		Variable dw1 = new Variable("dw1", new GlorotNormal(), 500, 10);
		Variable db1 = new Variable("db1", new Zeros(), 1, 10);
		Node output = new SoftmaxNode(new AdditionNode(new MmulNode(d0, dw1), db1));
		List<Variable> parameters = output.get_parameters();
		// only want weight parameters for L2 regularization
		Node loss = new AdditionNode(new CCENode(target, output), new L2PenaltyNode(w0, w1, dw0, dw1)); 
		// only one epoch though doesn't matter
		LearningRateScheduler scheduler = (epoch, lr) -> 0.001d * Math.pow(0.9d, lr);
		Optimizer optimizer = new Adam(0.001f, 0.9f, 0.999f, 0.0f, 1e-7f, false);
		println("finished initializing neural network");
		println("begin training");
		long start_training = System.nanoTime();
		loss.setTraining(true);
		for(int epoch = 1; epoch <= num_epochs; epoch++) {
			println("epoch " + epoch + " start");
			for(int i = 0; i < dataset.train_data.size(); i++) {
				input.set(dataset.train_data.get(i));
				target.set(dataset.train_labels.get(i));
				loss.evaluate();
				loss.compute_gradients();
				optimizer.update(parameters);
				loss.zero_grads();
				training_steps++;
				if(training_steps % 100 == 0) println(training_steps+" training steps completed");
				Nd4j.getMemoryManager().invokeGc();
			}
			optimizer.lr = (float) scheduler.apply(epoch, optimizer.lr);
			System.gc();
			println("epoch " + epoch + " end");
		}
		loss.setTraining(false);
		long end_training = System.nanoTime();
		double training_time = (end_training - start_training) / 1000000000.0d;
		println("finished training on 60000 samples in " + num_epochs + " epochs after " + training_time + " seconds");
		println("begin evaluation");
		// evaluate the network
		input.set(dataset.test_data.get(0));
		target.set(dataset.test_labels.get(0));
		loss.evaluate();
		int num_correct = output.m.argMax(-1).eq(target.m.argMax(-1)).castTo(Nd4j.defaultFloatingPointType()).sumNumber().intValue();
		loss.setEvaluated(false);
		Nd4j.getMemoryManager().invokeGc();
		double accuracy = num_correct/100.0d; // as a percentage
		println("accuracy on test data (10000 samples): " + accuracy + "% of test examples classified correctly");
		t.interrupt();
		println("end evaluation");
		println("end mnist dataset test session");
	}
	public static void println(Object x) {
		System.out.println(LocalDateTime.now().format(formatter) + " " + x);
	}
	public static void cmd() throws IOException {
		String str = br.readLine();
		if (str.equals("/steps")) println(training_steps+" training steps completed");
	}
}
