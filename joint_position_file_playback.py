#!/usr/bin/env python

import argparse
import sys

import rospy

import baxter_interface

from baxter_interface import CHECK_VERSION


def try_float(x):
    try:
        return float(x)
    except ValueError:
        return None


def clean_line(line, names):

    #convert the line of strings to a float or None
    line = [try_float(x) for x in line.rstrip().split(',')]
    #zip the values with the joint names
    combined = zip(names[1:], line[1:])
    #take out any tuples that have a none value
    cleaned = [x for x in combined if x[1] is not None]
    #convert it to a dictionary with only valid commands
    command = dict(cleaned)
    left_command = dict((key, command[key]) for key in command.keys()
                        if key[:-2] == 'left_')
    right_command = dict((key, command[key]) for key in command.keys()
                         if key[:-2] == 'right_')
    return (command, left_command, right_command, line)


def map_file(filename, loops=1):

    left = baxter_interface.Limb('left')
    right = baxter_interface.Limb('right')
    grip_left = baxter_interface.Gripper('left', CHECK_VERSION)
    grip_right = baxter_interface.Gripper('right', CHECK_VERSION)
    rate = rospy.Rate(1000)

    if grip_left.error():
        grip_left.reset()
    if grip_right.error():
        grip_right.reset()
    if (not grip_left.calibrated() and
        grip_left.type() != 'custom'):
        grip_left.calibrate()
    if (not grip_right.calibrated() and
        grip_right.type() != 'custom'):
        grip_right.calibrate()

    print("Playing back: %s" % (filename,))
    with open(filename, 'r') as f:
        lines = f.readlines()
    keys = lines[0].rstrip().split(',')

    l = 0
    # If specified, repeat the file playback 'loops' number of times
    while loops < 1 or l < loops:
        i = 0
        l += 1
        print("Moving to start position...")

        _cmd, lcmd_start, rcmd_start, _raw = clean_line(lines[1], keys)
        left.move_to_joint_positions(lcmd_start)
        right.move_to_joint_positions(rcmd_start)
        start_time = rospy.get_time()
        for values in lines[1:]:
            i += 1
            loopstr = str(loops) if loops > 0 else "forever"
            sys.stdout.write("\r Record %d of %d, loop %d of %s" %
                             (i, len(lines) - 1, l, loopstr))
            sys.stdout.flush()

            cmd, lcmd, rcmd, values = clean_line(values, keys)
            #command this set of commands until the next frame
            while (rospy.get_time() - start_time) < values[0]:
                if rospy.is_shutdown():
                    print("\n Aborting - ROS shutdown")
                    return False
                if len(lcmd):
                    left.set_joint_positions(lcmd)
                if len(rcmd):
                    right.set_joint_positions(rcmd)
                if ('left_gripper' in cmd and
                    grip_left.type() != 'custom'):
                    grip_left.command_position(cmd['left_gripper'])
                if ('right_gripper' in cmd and
                    grip_right.type() != 'custom'):
                    grip_right.command_position(cmd['right_gripper'])
                rate.sleep()
        print
    return True


def main():

    epilog = """
Related examples:
  joint_recorder.py; joint_trajectory_file_playback.py.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__,
                                     epilog=epilog)
    parser.add_argument(
        '-f', '--file', metavar='PATH', required=True,
        help='path to input file'
    )
    parser.add_argument(
        '-l', '--loops', type=int, default=1,
        help='number of times to loop the input file. 0=infinite.'
    )
    args = parser.parse_args(rospy.myargv()[1:])

    print("Initializing node... ")
    rospy.init_node("rsdk_joint_position_file_playback")
    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled

    def clean_shutdown():
        print("\nExiting example...")
        if not init_state:
            print("Disabling robot...")
            rs.disable()
    rospy.on_shutdown(clean_shutdown)

    print("Enabling robot... ")
    rs.enable()

    map_file(args.file, args.loops)


if __name__ == '__main__':
    main()
