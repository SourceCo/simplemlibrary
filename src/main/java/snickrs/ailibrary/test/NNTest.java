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
import snickrs.ailibrary.autodiff.normalization.*;
import snickrs.ailibrary.autodiff.optimization.*;
import snickrs.ailibrary.autodiff.regularization.*;
import snickrs.ailibrary.autodiff.reshape.*;
import snickrs.ailibrary.autodiff.rnn.*;
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
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class NNTest {
	public static DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm:ss.SSS");
	public static int training_steps = 0; // number of training steps so far
	public static float lr = 0.001f; // initial network learning rate;
	public static int num_epochs = 1; // how many times we iterate over the dataset
	public static int batch_size = 64; // how many examples per update --> less noisy training
	public static BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
	public static DecimalFormat decimal = new DecimalFormat("00.00");
	public static List<Variable> weights = new ArrayList<Variable>();
	public static void main(String[] args) throws FileNotFoundException, IOException {
		Nd4j.getRandom().setSeed(20240613); // random seed for reproducibility
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
		Mnist dataset = new Mnist(batch_size, true, true);
		dataset.squeeze_data();
		println("finished loading mnist dataset");
		Placeholder input = new Placeholder("input", dataset.input_dims);
		Placeholder target = new Placeholder("target", dataset.label_dims);

		Variable w0 = new Variable("w0", new GlorotUniform(), input.shape()[1], 100); // bc relu
		Node pa0 = new MmulNode(input, w0);
		Variable b0 = new Variable("b0", new Zeros(), pa0.shape());
		Node a0 = new LReluNode(new AdditionNode(pa0, b0));
		
		Variable w1 = new Variable("w1", new GlorotNormal(), a0.shape()[1], 10); // bc softmax
		Node pa1 = new MmulNode(a0, w1);
		Variable b1 = new Variable("b1", new Zeros(), pa1.shape());
		Node output = new SoftmaxNode(new AdditionNode(pa1, b1));
		
		weights.add(w0);
		weights.add(w1);
		
		List<Variable> parameters = output.get_parameters();
		Node loss = new CCENode(target, output);
		LearningRateScheduler scheduler = (epoch, lr) -> 0.001d * Math.pow(0.9d, lr);
		Optimizer optimizer = new Adam(lr, 0.9f, 0.999f, 0.0f, 1e-7f, false);
		println("begin training");
		long start_training = System.nanoTime();
		loss.setTraining(true);
		for(int epoch = 1; epoch <= num_epochs; epoch++) {
			String pct = "0.000%";
			print("epoch " + epoch + "/" + num_epochs + " --> " + pct);
			for(int i = 0; i < dataset.train_data.size(); i++) {
				input.set(dataset.train_data.get(i));
				target.set(dataset.train_labels.get(i));
				loss.evaluate();
				loss.compute_gradients();
				optimizer.update(parameters);
				loss.zero_grads();
				training_steps++;
				String x = "";
				for(int n = 0; n < pct.length(); n++) {
					x += "\b";
				}
				pct = decimal.format(100.0d*(i+1)/dataset.train_data.size())+"%";
				System.out.print(x+pct);
				Nd4j.getMemoryManager().invokeGc();
			}
			optimizer.lr = (float) scheduler.apply(epoch, optimizer.lr);
			if(epoch < num_epochs) {
				loss.setTraining(false);
				input.set(dataset.test_data.get(0));
				double accuracy = output.evaluate().argMax(-1).castTo(Nd4j.defaultFloatingPointType()).eq(dataset.test_key).castTo(Nd4j.defaultFloatingPointType()).sumNumber().intValue()/100.0d;
				output.setEvaluated(false);
				System.out.print(" --> accuracy on test data: " + accuracy + "%");
				loss.setTraining(true);
			}
			System.out.println();
			System.gc();
		}
		loss.setTraining(false);
		long end_training = System.nanoTime();
		double training_time = (end_training - start_training) / 1000000000.0d;
		println("finished training after " + training_time + " seconds");
		println("begin evaluation");
		// evaluate the network
		input.set(dataset.test_data.get(0));
		double accuracy = output.evaluate().argMax(-1).castTo(Nd4j.defaultFloatingPointType()).eq(dataset.test_key).castTo(Nd4j.defaultFloatingPointType()).sumNumber().intValue()/100.0d;
		output.setEvaluated(false);
		Nd4j.getMemoryManager().invokeGc();
		println("final accuracy on test data: " + accuracy + "%");
		System.exit(0);
	}
	public static Node mlp(Node input) {
		Variable w0 = new Variable("w0", new HeNormal(), input.shape()[1], 1); // bc relu
		Node pa0 = new MmulNode(input, w0);
		Variable b0 = new Variable("b0", new Zeros(), pa0.shape());
		Node a0 = new LReluNode(new AdditionNode(pa0, b0));
		
		Variable w1 = new Variable("w1", new GlorotNormal(), a0.shape()[1], 10); // bc softmax
		Node pa1 = new MmulNode(a0, w1);
		Variable b1 = new Variable("b1", new Zeros(), pa1.shape());
		Node output = new SoftmaxNode(new AdditionNode(pa1, b1));
		
//		Variable w2 = new Variable("w2", new GlorotNormal(), a1.shape()[1], 10); // bc softmax
//		Node pa2 = new MmulNode(a1, w2);
//		Variable b2 = new Variable("b2", new Zeros(), pa2.shape());
//		Node output = new SoftmaxNode(new AdditionNode(pa2, b2));
		
		weights.add(w0);
		weights.add(w1);
		return output;
	}
	public static Node lenet(Node input) {
		// create the LeNet model
		// layer 1 (convolutional)
		Variable w0 = new Variable("w0", new GlorotUniform(), 5, 5, 1, 20);
		Node c0 = new Conv2DNode(input, w0);
		Variable b0 = new Variable("b0", new Zeros(), c0.shape());
		Node a0 = new TanhNode(new AdditionNode(c0, b0));
		// layer 2 (pooling)
		Node p0 = new MaxPooling2DNode(a0);
		// layer 3 (convolutional)
		Variable w1 = new Variable("w1", new GlorotUniform(), 5, 5, 20, 50);
		Node c1 = new Conv2DNode(p0, w1);
		Variable b1 = new Variable("b1", new Zeros(), c1.shape());
		Node a1 = new TanhNode(new AdditionNode(c1, b1));
		// layer 4 (pooling)
		Node p1 = new MaxPooling2DNode(a1);
		// layer 5 (dense)
		Node flat = new FlattenNode(p1);
		Variable dw0 = new Variable("dw0", new GlorotNormal(), flat.shape()[1], 500);
		Variable db0 = new Variable("db0", new Zeros(), 1, 500);
		Node d0 = new TanhNode(new AdditionNode(new MmulNode(flat, dw0), db0));
		// layer 6 (dense)
		Variable dw1 = new Variable("dw1", new GlorotNormal(), 500, 10);
		Variable db1 = new Variable("db1", new Zeros(), 1, 10);
		weights.add(w0);
		weights.add(w1);
		weights.add(dw0);
		weights.add(dw1);
		Node output = new SoftmaxNode(new AdditionNode(new MmulNode(d0, dw1), db1));
		return output;
	}
	public static Node cnn2d(Node input) {
//		this is a simple 2d cnn
//		after only 1 epoch, reachs 96.87% accuracy as well on test data
		Variable w0 = new Variable("w0", new GlorotUniform(), 3, 3, 1, 8);
		Node c0 = new Conv2DNode(input, w0);
		Variable b0 = new Variable("b0", new Zeros(), 1, 8, 26, 26);
		Node a0 = new LReluNode(new AdditionNode(c0, b0));
		Node p0 = new MaxPooling2DNode(a0);
		Variable w1 = new Variable("w1", new GlorotUniform(), 3, 3, 8, 16);
		Node c1 = new Conv2DNode(p0, w1);
		Variable b1 = new Variable("b1", new Zeros(), 1, 16, 11, 11);
		Node a1 = new LReluNode(new AdditionNode(c1, b1));
		Node p1 = new MaxPooling2DNode(a1);
		Node flat = new FlattenNode(p1);
		Variable dw0 = new Variable("dw0", new HeNormal(), flat.shape()[1], 64);
		Variable db0 = new Variable("dw0", new Zeros(), 1, 64);
		Node d0 = new LReluNode(new AdditionNode(new MmulNode(flat, dw0), db0));
		Variable dw1 = new Variable("dw1", new GlorotNormal(), 64, 10);
		Variable db1 = new Variable("db1", new Zeros(), 1, 10);
		weights.add(w0);
		weights.add(w1);
		weights.add(dw0);
		weights.add(dw1);
		Node output = new SoftmaxNode(new AdditionNode(new MmulNode(d0, dw1), db1));
		return output;
	}
	public static Node cnn1d(Node input) {
//		an experimental 1d cnn
		Variable w0 = new Variable("w0", new GlorotUniform(), 81, 1, 4);
		Node c0 = new Conv1DNode(input, w0);
		Variable b0 = new Variable("b0", new Zeros(), c0.shape());
		Node a0 = new LReluNode(new AdditionNode(c0, b0));
		Node p0 = new MaxPooling1DNode(a0);
		Variable w1 = new Variable("w1", new GlorotUniform(), 81, 4, 4);
		Node c1 = new Conv1DNode(p0, w1);
		Variable b1 = new Variable("b1", new Zeros(), c1.shape());
		Node a1 = new LReluNode(new AdditionNode(c1, b1));
		Node p1 = new MaxPooling1DNode(a1);
		Node flat = new FlattenNode(p1);
		Variable dw0 = new Variable("dw0", new HeNormal(), flat.shape()[1], 64);
		Variable db0 = new Variable("dw0", new Zeros(), 1, 64);
		Node d0 = new LReluNode(new AdditionNode(new MmulNode(flat, dw0), db0));
		Variable dw1 = new Variable("dw1", new GlorotNormal(), 64, 10);
		Variable db1 = new Variable("db1", new Zeros(), 1, 10);
		weights.add(w0);
		weights.add(w1);
		weights.add(dw0);
		weights.add(dw1);
		Node output = new SoftmaxNode(new AdditionNode(new MmulNode(d0, dw1), db1));
		return output;
	}
	public static void println(Object x) {
		System.out.println(LocalDateTime.now().format(formatter) + " " + x);
	}
	public static void print(Object x) {
		System.out.print(LocalDateTime.now().format(formatter) + " " + x);
	}
	public static void cmd() throws IOException {
		String str = br.readLine();
		if (str.equals("/steps")) println(training_steps+" training steps completed");
	}
}
