package snickrs.ailibrary.datasets;
import java.awt.*;
import java.awt.event.*;
import java.io.FileNotFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;

public class MnistDisplay extends Frame {
	private static final long serialVersionUID = 3791164113495222068L;
	public Mnist dataset;
	public int width;
	public int height;
	public int box_width;
	public int box_height;
	public MnistDisplay() throws FileNotFoundException {
		this.dataset = new Mnist(1, false, false);
		this.width = 600;
		this.height = 600;
		this.box_width = 20;
		this.box_height = 20;
		setVisible(true);
		setTitle("mnist");
		setSize(width, height);
		addWindowListener(new WindowAdapter() { 
            @Override
            public void windowClosing(WindowEvent e) 
            { 
                System.exit(0); 
            } 
        }); 
		setBackground(new Color(255, 255, 255));
	}
	@Override
	public void paint(Graphics g) {
		super.paint(g);
	}
	public void render(int idx) {
		Graphics2D g = (Graphics2D) getGraphics();
		INDArray example = dataset.train_data.get(idx);
		for(int i = 0; i < 28; i++) {
			for(int j = 0; j < 28; j++) {
				int opacity = example.getInt(1, 1, i, j);
				if(opacity == 0) continue;
				g.setColor(new Color(255, 0, 0, opacity));
				g.fillRect(i*box_width, j*box_height, box_width, box_height);
			}
		}
	}
}
