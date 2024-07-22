package snickrs.ailibrary.datasets;

import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.io.*;
import java.util.*;

public class Mnist {
	public List<INDArray> train_data;
	public List<INDArray> train_labels;
	public List<INDArray> test_data;
	public List<INDArray> test_labels;
	public INDArray test_key;
	public long[] input_dims;
	public long[] label_dims;
	public static final String train_path = "C:\\Users\\saraa\\eclipse-workspace\\ailibrary\\mnist_dataset\\mnist_train.txt";
	public static final String test_path = "C:\\Users\\saraa\\eclipse-workspace\\ailibrary\\mnist_dataset\\mnist_test.txt";
	public File train_file;
	public File test_file;
	public int batch_size;
	public boolean flatten;
	public Mnist(int batch_size, boolean flatten, boolean normalize) throws FileNotFoundException {
		init_data(batch_size, flatten, normalize);
	}
	public Mnist(int batch_size, boolean flatten) throws FileNotFoundException {
		init_data(batch_size, flatten, true);
	}
	public Mnist() throws FileNotFoundException {
		init_data(1, true, true);
	}
	public void init_data(int batch_size, boolean flatten, boolean normalize) throws FileNotFoundException {
		this.train_file = new File(train_path);
		this.test_file = new File(test_path);
		this.train_data = new ArrayList<INDArray>();
		this.train_labels = new ArrayList<INDArray>();
		this.test_data = new ArrayList<INDArray>();
		this.test_labels = new ArrayList<INDArray>();
		this.test_key = test_key();
		this.batch_size = batch_size;
		this.flatten = flatten;
		this.input_dims = flatten ? new long[] {1, 1, 784} : new long[] {1, 1, 28, 28};
		this.label_dims = new long[] {1, 10};
		generate_data(train_file, train_data, train_labels, flatten, normalize, batch_size);
		generate_data(test_file, test_data, test_labels, flatten, normalize, 10000); // this is how many test samples there are
	}
//	public void generate_training_data() throws FileNotFoundException {
//		Scanner scanner = new Scanner(train_file);
//		while(scanner.hasNext()) {
//			String[] line = scanner.nextLine().split(",");
//			INDArray label = Nd4j.zeros(1, 10);
//			label.putScalar(Integer.valueOf(line[0]), 1.0f);
//			float[][] data = new float[1][line.length-1];
//			for(int i = 1; i < line.length-1; i++) {
//				data[0][i] = Float.valueOf(line[i]);
//			}
//			train_data.add(Nd4j.create(data).div(255.0f));
//			train_labels.add(label);
//		}
//		scanner.close();
//	}
//	public void generate_test_data() throws FileNotFoundException {
//		Scanner scanner = new Scanner(test_file);
//		while(scanner.hasNext()) {
//			String[] line = scanner.nextLine().split(",");
//			INDArray label = Nd4j.zeros(1, 10);
//			label.putScalar(Integer.valueOf(line[0]), 1.0f);
//			float[][] data = new float[1][line.length-1];
//			for(int i = 1; i < line.length-1; i++) {
//				data[0][i] = Float.valueOf(line[i]);
//			}
//			test_data.add(Nd4j.create(data).div(255.0f));
//			test_labels.add(label);
//		}
//		scanner.close();
//	}
	public INDArray test_key() throws FileNotFoundException {
		INDArray test_key = Nd4j.zeros(10000);
		Scanner scanner = new Scanner(this.test_file);
		scanner.useDelimiter("\n");
		int i = 0;
		while(scanner.hasNext()) {
			test_key.putScalar(i++, Integer.parseInt(Character.toString(scanner.next().charAt(0))));
		}
		scanner.close();
		return test_key;
	}
	public void generate_data(File file, List<INDArray> arr_data, List<INDArray> arr_labels, boolean flatten, boolean normalize, int batch_size) throws FileNotFoundException {
		Scanner scanner = new Scanner(file);
		scanner.useDelimiter("\n");
		while(scanner.hasNext()) {
			INDArray label = Nd4j.zeros(batch_size, 10);
			INDArray data = Nd4j.create(batch_size, 1, 784);
			String[][] text = new String[batch_size][785];
			for(int i = 0; i < text.length; i++) {
				if(!scanner.hasNext()) break;
				text[i] = scanner.next().split(",");
				label.putScalar(new int[] {i, Integer.valueOf(text[i][0])}, 1.0d);
				for(int j = 0; j < text[i].length-1; j++) {
					int val = Integer.valueOf(text[i][j+1]);
					data.putScalar(new int[] {i, 0, j}, normalize ? val/255.0d : val);
				}
			}
			arr_data.add(flatten ? data : data.reshape(batch_size, 1, 28, 28).permute(0, 1, 3, 2)); // permute because height and width are switched
			arr_labels.add(label);
		}
		scanner.close();
	}
	public void squeeze_data() {
		this.input_dims = remove(this.input_dims, 1);
		for(int i = 0; i < train_data.size(); i++) {
			train_data.set(i, Nd4j.squeeze(train_data.get(i), 1));
		}
		for(int i = 0; i < test_data.size(); i++) {
			test_data.set(i, Nd4j.squeeze(test_data.get(i), 1));
		}
		Nd4j.getMemoryManager().invokeGc();
	}
    public long[] remove(long[] array, int index) {
        if (index < 0) return remove(array, array.length+index);
        if (index >= array.length) return array;
        long[] new_arr = new long[array.length - 1];
        for (int i = 0, j = 0; i < array.length; i++) {
            if (i == index) continue;
            new_arr[j++] = array[i];
        }
        return new_arr;
    }
	// slow and inefficient
//	public void get_data(File file, List<INDArray> arr_data, List<INDArray> arr_labels, int batch_size) throws FileNotFoundException {
//		Scanner scanner = new Scanner(file);
//		scanner.useDelimiter("[,\n]");
//		while(scanner.hasNext()) {
//			INDArray label = Nd4j.zeros(batch_size, 10);
//			INDArray data = Nd4j.create(batch_size, 784);
//			for(int line = 0; line < batch_size; line++) {
//				if(!scanner.hasNext()) break;
//				label.putScalar(line*10+scanner.nextInt(), 1.0d);
//				for(int i = 0; i < 784; i++) {
//					data.putScalar(line*784+i, scanner.nextInt()/255.0d);
//				}
//			}
//			arr_data.add(this.flatten ? data : data.reshape(batch_size, 28, 28));
//			arr_labels.add(label);
//		}
//		scanner.close();
//	}
}
