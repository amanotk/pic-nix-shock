#!/usr/bin/env python

import os
import sys

import numpy as np
import toml


def read_and_print_config(dirname, time=None):
    config = os.path.join(dirname, "config.toml")

    if os.path.exists(config) and os.path.isfile(config):
        print(f"***** {dirname} *****")
        try:
            with open(config, "r", encoding="utf-8") as f:
                content = f.read()
                cfgdata = toml.loads(content)
                maxstep = summarize_config(cfgdata, time)
                estimate_output(cfgdata, maxstep)
        except Exception as e:
            print(f"Failed to read {config}: {e}")
    else:
        print(f"No config.toml found in {dirname}")


def summarize_config(obj, time):
    application = obj.get("application", None)
    parameter = obj.get("parameter", None)
    if application is None or parameter is None:
        print("Invalid configuration file")
        return

    order = application["option"].get("order", 2)

    Nx = parameter["Nx"]
    Ny = parameter["Ny"]
    Nz = parameter["Nz"]
    Cx = parameter["Cx"]
    Cy = parameter["Cy"]
    Cz = parameter["Cz"]
    u0 = parameter["u0"]
    mime = parameter["mime"]
    sigma = parameter["sigma"]
    theta = parameter["theta"]
    phi = parameter["phi"]
    betai = parameter["betai"]
    betae = parameter["betae"]
    delt = parameter["delt"]
    delh = parameter["delh"]

    if Nx % Cx != 0 or Ny % Cy != 0 or Nz % Cz != 0:
        print("Number of grid must be divisible by number of chunk.")
        return

    Mx = Nx // Cx
    My = Ny // Cy
    Mz = Nz // Cz
    vae = np.sqrt(sigma)
    vai = np.sqrt(sigma / mime)
    vte = vae * np.sqrt(0.5 * betae)
    vti = vai * np.sqrt(0.5 * betai)
    Ma0 = u0 / vai
    wce = np.sqrt(sigma)
    wci = wce / mime
    rgi = u0 / wci
    Tshock = int(1 / (wci * delt))
    Lshock = int(rgi / delh)
    maxstep = int(time / (wci * delt))

    print("{:30s} : {:5d} x {:5d} x {:5d}".format("Number of Grid", Nz, Ny, Nx))
    print("{:30s} : {:5d} x {:5d} x {:5d}".format("Number of Chunk", Cz, Cy, Cx))
    print("{:30s} : {:5d} x {:5d} x {:5d}".format("Chunk Dimension", Mz, My, Mx))
    print("{:30s} : {:12d}".format("Order of shape function", order))
    print("{:30s} : {:12d}".format("Number of Step", maxstep))
    print("{:30s} : {:12d}".format("Number of step for Tshock", Tshock))
    print("{:30s} : {:12d}".format("Number of grid for Lshock", Lshock))
    print("{:30s} : {:12.3e}".format("Ma0", Ma0))
    print("{:30s} : {:12.3e}".format("theta", theta))
    print("{:30s} : {:12.3e}".format("phi", phi))
    print("{:30s} : {:12.3e}".format("u0", u0))
    print("{:30s} : {:12.3e}".format("vae", vae))
    print("{:30s} : {:12.3e}".format("vte", vte))
    print("{:30s} : {:12.3e}".format("vai", vai))
    print("{:30s} : {:12.3e}".format("vti", vti))
    print("{:30s} : {:12.3e}".format("mime", mime))

    return maxstep


def estimate_output(obj, maxstep):
    diagnostic = obj.get("diagnostic", None)
    parameter = obj.get("parameter", None)
    if diagnostic is None or parameter is None:
        print("Invalid configuration file")
        return

    field_size = 0.0
    particle_size = 0.0
    for diag in diagnostic:
        if diag["name"] == "field":
            field_size += esimate_output_field(diag, parameter, maxstep)
        if diag["name"] == "particle":
            particle_size += esimate_output_particle(diag, parameter, maxstep)

    total_size = field_size + particle_size
    print(
        "{:30s} : {:8.2e} (field = {:8.2e}, particle = {:8.2e})".format(
            "Estimated output [GB]", total_size, field_size, particle_size
        )
    )


def esimate_output_field(diag, parameter, maxstep):
    num_field = 6
    num_moment = 14
    Ns = parameter["Ns"]
    Nx = parameter["Nx"]
    Ny = parameter["Ny"]
    Nz = parameter["Nz"]

    elemsize = (num_field + num_moment * Ns) * 8
    gigabyte = 1024 * 1024 * 1024

    ## size for each snapshot
    decimate = diag.get("decimate", 1)
    Mx = Nx // decimate if Nx % decimate == 0 else Nx
    My = Ny // decimate if Ny % decimate == 0 else Ny
    Mz = Nz // decimate if Nz % decimate == 0 else Nz
    snapshot_size = elemsize * Mx * My * Mz

    ## number of snapshots
    begin = diag.get("begin", 0)
    end = diag.get("end", maxstep)
    interval = diag.get("interval", 1)
    num_snapshots = (end - begin) // interval + 1

    return snapshot_size * num_snapshots / gigabyte


def esimate_output_particle(diag, parameter, maxstep):
    nppc = parameter["nppc"]
    Ns = parameter["Ns"]
    Nx = parameter["Nx"]
    Ny = parameter["Ny"]
    Nz = parameter["Nz"]
    u0 = parameter["u0"]
    delt = parameter["delt"]
    delh = parameter["delh"]
    vinj = u0 / np.sqrt(1 + u0 * u0)

    elemsize = 7 * 8  # 7 fields per particle
    fraction = diag.get("fraction", 1.0)
    gigabyte = 1024 * 1024 * 1024

    begin = diag.get("begin", 0)
    end = diag.get("end", maxstep)
    interval = diag.get("interval", 1)

    # take care of the newly injected particles
    total_size = 0.0
    for step in range(begin, end + 1, interval):
        Nini = nppc * Nx * Ny * Nz * Ns
        Ninj = nppc * vinj * delt / delh * Ny * Nz * Ns * step
        Ntot = Nini + Ninj
        size = Ntot * fraction * elemsize / gigabyte
        total_size += size

    return total_size


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-t", "--time", type=float, default=None, help="time in unit of ion gyrotime"
    )
    parser.add_argument("dirs", nargs="*", help="directories to process")
    args = parser.parse_args()

    # check required arguments
    if args.time is None:
        parser.error("The following arguments are required: -t/--time")
        sys.exit(1)

    return args


def main(dirs, time=None):
    for dirname in dirs:
        if os.path.isdir(dirname):
            read_and_print_config(dirname, time)


def cli_main():
    args = parse_args()
    main(args.dirs, args.time)


if __name__ == "__main__":
    cli_main()
