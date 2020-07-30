import matplotlib.pyplot as plt
import numpy
import os

from mpl_toolkits.mplot3d.art3d import Line3DCollection
import struct



def vis_point(test_loader,  output_path, output_path_error, prob, y_pred_list, y_real_list, crit_points_list_ind=None):

    if crit_points_list_ind is not None:
        vis_crit_points(test_loader,  output_path, output_path_error, prob, y_pred_list, y_real_list, crit_points_list_ind)
        return

    for i, data in enumerate(test_loader):

        if i == 3:
            break

        certainty = prob[i]
        y_real = y_real_list[i]
        y_pred = y_pred_list[i]
        y_real_l = test_loader.test_data.classmap[y_real]
        y_pred_l = test_loader.test_data.classmap[y_pred]
        pos = data.pos.numpy()

        xyz = numpy.array(
            [list(a) for a in zip(data.pos[:, 0].numpy(), data.pos[:, 1].numpy(), data.pos[:, 2].numpy())])

        ax = plt.axes(projection='3d')
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color='black', s=10)

        ax.set_xlim(left=1, right=-1)
        ax.set_ylim(bottom=1, top=-1)
        ax.set_zlim(-1, 1)

        if y_pred != y_real:
            out = output_path_error + "/" + str(i) + y_real_l + "-" + y_pred_l + "-with(" + str(certainty) + ")"
            with open(out + ".txt", "w") as text_file:
                for line in pos:
                    text_file.write(str(line[0]) + ', ' + str(line[1]) + ', ' + str(line[2]) + '\n')
            plt.savefig(out + '.png')
            plt.show()
        else:
            out = output_path + "/" + str(i) + y_real_l + "-" + y_pred_l + "-with(" + str(certainty) + ")"
            with open(out + ".txt", "w") as text_file:
                for line in pos:
                    text_file.write(str(line[0]) + ', ' + str(line[1]) + ', ' + str(line[2]) + '\n')
            plt.savefig(out + '.png')
            plt.show()


def vis_crit_points(test_loader,  output_path, output_path_error, prob, y_pred_list, y_real_list, crit_points_list_ind=None):

    for i, data in enumerate(test_loader):

        """if i == 3:
            break"""

        certainty = prob[i]
        y_real = y_real_list[i]
        y_pred = y_pred_list[i]
        y_real_l = test_loader.dataset.classmap[y_real]
        y_pred_l = test_loader.dataset.classmap[y_pred]
        pos = data.pos.numpy()

        xyz = numpy.array(
            [list(a) for a in zip(data.pos[:, 0].numpy(), data.pos[:, 1].numpy(), data.pos[:, 2].numpy())])

        crit_unique_ind = numpy.unique(crit_points_list_ind[i])
        crit_points = numpy.vstack([xyz[j] for j in crit_unique_ind])
        # print("Shown {} critical points".format(len(crit_points)))

        fig = plt.figure(figsize=plt.figaspect(0.5))
        fig.suptitle('True label: {} / Predicted label: {}' .format(y_real_l, y_pred_l), fontsize=16)

        # full pointcloud
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color='black', s=10)
        ax.set_xlim(left=1, right=-1)
        ax.set_ylim(bottom=1, top=-1)
        ax.set_zlim(-1, 1)
        ax.title.set_text("full pointcloud")

        # critical point
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(crit_points[:, 0], crit_points[:, 1], crit_points[:, 2], color='red', s=20)
        ax.set_xlim(left=1, right=-1)
        ax.set_ylim(bottom=1, top=-1)
        ax.set_zlim(-1, 1)
        ax.title.set_text("critical points")

        if y_pred != y_real:
            out = output_path_error + "/" + str(i) + y_real_l + "-" + y_pred_l + "-with(" + str(certainty) + ")"
            with open(out + ".txt", "w") as text_file:
                for line in pos:
                    text_file.write(str(line[0]) + ', ' + str(line[1]) + ', ' + str(line[2]) + '\n')
            plt.savefig(out + '.png')
            plt.show()
        else:
            out = output_path + "/" + str(i) + y_real_l + "-" + y_pred_l + "-with(" + str(certainty) + ")"
            with open(out + ".txt", "w") as text_file:
                for line in pos:
                    text_file.write(str(line[0]) + ', ' + str(line[1]) + ', ' + str(line[2]) + '\n')
            plt.savefig(out + '.png')
            plt.show()

def vis_graph(loader, out_path):

    limit = 3

    for i, data in enumerate(loader):
        l = data.y.item()
        label = loader.dataset.classmap[l]
        X = data.pos[:, 0].numpy()
        Y = data.pos[:, 1].numpy()
        Z = data.pos[:, 2].numpy()
        xyzn = list(zip(X, Y, Z))
        enlarged_label = numpy.repeat(l, len(X))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(X, Y, Z, c=enlarged_label, depthshade=True, marker='o', s=2)
        ax.set_xlim(left=1, right=-1)
        ax.set_ylim(bottom=1, top=-1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("instance label " + str(label))

        # if data.edge_index != None:
        indexl = data.edge_index[0]
        indexr = data.edge_index[1]

        edges_test = []
        for a, b in zip(indexl.numpy(), indexr.numpy()):
            edges_test.append((a, b))

        segments = [(xyzn[s], xyzn[t]) for s, t in edges_test]
        edge_col = Line3DCollection(segments, lw=0.5, colors='b')
        ax.add_collection3d(edge_col)
        #plt.show()


        filename = str(label) + str(i) +'.png'
        path = os.path.join(out_path, filename)
        plt.savefig(path)





#def write_pointcloud(filename,xyz_points,rgb_points=None):
def write_pointcloud(loader, out_path, rgb_points=None, xyz_points=None):

    """ creates a .pkl file of the point clouds generated
    """
    for i, data in enumerate(loader):
        l = data.y.item()
        label = loader.dataset.classmap[l]
        X = data.pos[:, 0].numpy()
        Y = data.pos[:, 1].numpy()
        Z = data.pos[:, 2].numpy()

        Xn = data.norm[:, 0].numpy()
        Yn = data.norm[:, 1].numpy()
        Zn = data.norm[:, 2].numpy()

        xyzn = list(zip(Xn, Yn, Zn))
        xyzn = numpy.array(xyzn)
        xyz = list(zip(X, Y, Z))
        xyz=numpy.array(xyz)
        filename = str(label) + str(i) + '.ply'
        path = os.path.join(out_path, filename)
        assert xyz.shape[1] == 3,'Input XYZ points should be Nx3 float array'
        if rgb_points is None:
            rgb_points = numpy.ones(xyz.shape).astype(numpy.uint8)*255
        assert xyz.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

        # Write header of .ply file
        fid = open(path,'wb')
        fid.write(bytes('ply\n', 'utf-8'))
        fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
        fid.write(bytes('element vertex %d\n'%xyz.shape[0], 'utf-8'))
        fid.write(bytes('property float x\n', 'utf-8'))
        fid.write(bytes('property float y\n', 'utf-8'))
        fid.write(bytes('property float z\n', 'utf-8'))
        fid.write(bytes('property uchar red\n', 'utf-8'))
        fid.write(bytes('property uchar green\n', 'utf-8'))
        fid.write(bytes('property uchar blue\n', 'utf-8'))
        fid.write(bytes('property float xn\n', 'utf-8'))
        fid.write(bytes('property float yn\n', 'utf-8'))
        fid.write(bytes('property float zn\n', 'utf-8'))

        fid.write(bytes('end_header\n', 'utf-8'))

        # Write 3D points to .ply file
        for i in range(xyz.shape[0]):
            fid.write(bytearray(struct.pack("ffffffccc",xyz[i,0],xyz[i,1],xyz[i,2],xyzn[i,0],xyzn[i,1],xyzn[i,2],
                                            rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                            rgb_points[i,2].tostring())))
        fid.close()



