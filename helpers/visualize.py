import matplotlib.pyplot as plt
import numpy




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

        if i == 3:
            break

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

def vis_graph_():
    s=2