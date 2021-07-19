import h5py
import plotly.offline as of
import plotly.graph_objs as go
of.offline.init_notebook_mode(connected=True)

def show_img():
    with h5py.File("archive/train_point_clouds.h5", "r") as points_dataset:        
        digits = []
        for i in range(10):
            digit = (points_dataset[str(i)]["img"][:], 
                points_dataset[str(i)]["points"][:], 
                points_dataset[str(i)].attrs["label"]) 
            digits.append(digit)
        
        x_c = [r[0] for r in digits[0][1]]
        y_c = [r[1] for r in digits[0][1]]
        z_c = [r[2] for r in digits[0][1]]
        trace1 = go.Scatter3d(x=x_c, y=y_c, z=z_c, mode='markers', 
                marker=dict(size=12, color=z_c, colorscale='Viridis', opacity=0.7))

    data = [trace1]
    layout = go.Layout(height=500, width=600, title= "Digit: "+str(digits[0][2]) + " in 3D space")
    fig = go.Figure(data=data, layout=layout)
    of.plot(fig)