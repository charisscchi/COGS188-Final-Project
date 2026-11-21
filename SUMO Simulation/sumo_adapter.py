"""Run a pinned SUMO scenario and convert its XML output to pickup measurements."""

import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import sumo
from sumolib.net import readNet

ROOT = Path(__file__).resolve().parent
OSM = ROOT / "data" / "lower-manhattan.osm.xml"
NETWORK = ROOT / "data" / "lower-manhattan.net.xml"
ROUTING = ROOT.parent / "Routing Algorithms"
sys.path.insert(0, str(ROUTING))

from road_network import Road, RoadNetwork, Turn
from synthetic_environment import PickupOutcome

VEHICLE_CLASS = "passenger"
STEP_S = 0.5


def binary(app):
    suffix = ".exe" if os.name == "nt" else ""
    return Path(sumo.__file__).resolve().parent / "bin" / (app + suffix)


def run_binary(app, *args, timeout=60):
    process = subprocess.run(
        [str(binary(app)), *map(str, args)], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout, cwd=ROOT,
    )
    if process.returncode:
        raise RuntimeError(f"{app} exited {process.returncode}\n{process.stderr}\n{process.stdout}")
    return process


def build_network():
    run_binary(
        "netconvert", "--osm-files", OSM.relative_to(ROOT),
        "--output-file", NETWORK.relative_to(ROOT), "--output.street-names",
        "--geometry.remove", "--tls.guess-signals", "--tls.join",
        "--keep-edges.by-vclass", VEHICLE_CLASS, "--no-turnarounds",
        "--xml-validation", "never", "--xml-validation.net", "never",
    )


def read_network():
    network = readNet(str(NETWORK), withInternal=True)
    roads = {}
    allowed_lanes = {}
    internal = {}
    permissions = frozenset({VEHICLE_CLASS})
    for edge in network.getEdges(withInternal=True):
        lanes = tuple(lane for lane in edge.getLanes() if lane.allows(VEHICLE_CLASS))
        if edge.getFunction() == "internal":
            internal.update({lane.getID(): lane for lane in edge.getLanes()})
        elif edge.getFunction() == "" and lanes:
            edge_id = edge.getID()
            allowed_lanes[edge_id] = lanes
            roads[edge_id] = Road(
                edge_id, edge.getFromNode().getID(), edge.getToNode().getID(),
                edge.getLength(), max(lane.getSpeed() for lane in lanes), permissions,
            )

    turns = []
    for edge_id in roads:
        edge = network.getEdge(edge_id)
        for next_edge, connections in edge.getOutgoing().items():
            next_id = next_edge.getID()
            if next_id not in roads:
                continue
            if any(
                connection.getFromLane() in allowed_lanes[edge_id]
                and connection.getToLane() in allowed_lanes[next_id]
                and (not connection.getViaLaneID()
                     or (connection.getViaLaneID() in internal
                         and internal[connection.getViaLaneID()].allows(VEHICLE_CLASS)))
                for connection in connections
            ):
                turns.append(Turn(edge_id, next_id, permissions))

    return network, RoadNetwork(roads.values(), turns), allowed_lanes


@dataclass(frozen=True)
class Measurement:
    outcome: PickupOutcome
    depart_s: float | None = None
    arrival_s: float | None = None
    arrival_lane: str | None = None
    arrival_position_m: float | None = None
    stopped_in_traffic_s: float | None = None
    distance_m: float | None = None


def write_xml(path, root):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


class SumoDriver:
    def __init__(self, lanes):
        self.lanes = lanes

    def drive_in_sumo(self, scenario, route, replay_dir=None):
        if route is None:
            return Measurement(PickupOutcome("unreachable"))
        if route[0] != scenario.start_edge or route[-1] != scenario.pickup_edge:
            raise ValueError(f"Route endpoints do not match {scenario.id}")
        if replay_dir is not None:
            replay_dir.mkdir(parents=True, exist_ok=True)
        location = (nullcontext(replay_dir) if replay_dir is not None
                    else TemporaryDirectory(dir=ROOT, prefix=".sumo-run-"))
        with location as folder:
            return self._run(Path(folder), scenario, route)

    def _run(self, folder, scenario, route):
        start_lane, = self.lanes[scenario.start_edge]
        pickup_lane, = self.lanes[scenario.pickup_edge]
        routes = ET.Element("routes")
        ET.SubElement(routes, "vType", id="car", vClass=VEHICLE_CLASS, sigma="0",
                      speedFactor="normc(1,0,0.2,2)")
        vehicles = [(departure, f"background-{index}", path)
                    for index, (departure, path) in enumerate(scenario.background)]
        vehicles.append((scenario.request_s, "taxi", route))
        for depart, vehicle_id, edges in sorted(vehicles):
            attributes = {"id": vehicle_id, "type": "car", "depart": f"{depart:.1f}", "departLane": "best"}
            if vehicle_id == "taxi":
                attributes.update(
                    departLane=str(start_lane.getIndex()), departPos=str(scenario.start_m),
                    arrivalLane=str(pickup_lane.getIndex()), arrivalPos=str(scenario.pickup_m),
                )
            vehicle = ET.SubElement(routes, "vehicle", attributes)
            ET.SubElement(vehicle, "route", edges=" ".join(edges))
        write_xml(folder / "routes.rou.xml", routes)

        additional = ET.Element("additional")
        for index, edge_id in enumerate(scenario.bottleneck_edges):
            for lane in self.lanes[edge_id]:
                sign = ET.SubElement(additional, "variableSpeedSign", id=f"cap-{index}-{lane.getIndex()}",
                                     lanes=lane.getID())
                ET.SubElement(sign, "step", time="0", speed=f"{lane.getSpeed() * scenario.actual_factor:.6f}")
        write_xml(folder / "traffic.add.xml", additional)

        config = ET.Element("configuration")
        inputs = ET.SubElement(config, "input")
        network_relative = Path(os.path.relpath(NETWORK, folder)).as_posix()
        ET.SubElement(inputs, "net-file", value=network_relative)
        ET.SubElement(inputs, "route-files", value="routes.rou.xml")
        ET.SubElement(inputs, "additional-files", value="traffic.add.xml")
        time = ET.SubElement(config, "time")
        deadline = scenario.request_s + scenario.timeout_s
        ET.SubElement(time, "end", value=str(deadline + STEP_S))
        ET.SubElement(time, "step-length", value=str(STEP_S))
        processing = ET.SubElement(config, "processing")
        ET.SubElement(processing, "time-to-teleport", value="-1")
        ET.SubElement(processing, "collision.action", value="warn")
        random = ET.SubElement(config, "random_number")
        ET.SubElement(random, "seed", value=str(scenario.seed))
        output = ET.SubElement(config, "output")
        ET.SubElement(output, "tripinfo-output", value="tripinfo.xml")
        ET.SubElement(output, "tripinfo-output.write-unfinished", value="true")
        report = ET.SubElement(config, "report")
        ET.SubElement(report, "no-step-log", value="true")
        ET.SubElement(report, "xml-validation", value="never")
        ET.SubElement(report, "xml-validation.net", value="never")
        config_path = folder / "scenario.sumocfg"
        write_xml(config_path, config)

        process = run_binary("sumo", "-c", config_path, timeout=35)
        log = process.stderr + process.stdout
        if "collision" in log.lower() or "teleporting" in log.lower():
            raise RuntimeError(f"Unexpected SUMO collision or teleport for {scenario.id}:\n{log}")

        trip = ET.parse(folder / "tripinfo.xml").find("./tripinfo[@id='taxi']")
        if trip is None:
            return Measurement(PickupOutcome("timeout"))
        depart = float(trip.attrib["depart"])
        if depart < 0:
            depart = None
        elif (trip.attrib["departLane"] != start_lane.getID()
              or abs(float(trip.attrib["departPos"]) - scenario.start_m) > 0.02):
            raise RuntimeError(f"Taxi departed from the wrong position in {scenario.id}")
        arrival = float(trip.attrib["arrival"])
        if arrival < 0 or arrival > deadline:
            return Measurement(PickupOutcome("timeout"), depart_s=depart)
        if depart is None or not scenario.request_s <= depart <= arrival:
            raise RuntimeError(f"Inconsistent SUMO timestamps in {scenario.id}")
        if trip.attrib.get("vaporized"):
            raise RuntimeError(f"Taxi was removed by SUMO in {scenario.id}")
        lane = trip.attrib["arrivalLane"]
        position = float(trip.attrib["arrivalPos"])
        # SUMO can report the position at the end of its last simulation step.
        max_overshoot = pickup_lane.getSpeed() * STEP_S + 0.02
        if lane != pickup_lane.getID() or not scenario.pickup_m - 0.02 <= position <= scenario.pickup_m + max_overshoot:
            raise RuntimeError(f"Taxi missed pickup in {scenario.id}: {lane} at {position} m")
        return Measurement(
            PickupOutcome("completed", arrival - scenario.request_s, arrival - depart),
            depart, arrival, lane, position, float(trip.attrib["waitingTime"]),
            float(trip.attrib["routeLength"]),
        )


if __name__ == "__main__":
    build_network()
    _, roads, _ = read_network()
    turn_count = sum(len(turns) for turns in roads.outgoing.values())
    print(f"{len(roads.roads)} passenger edges and {turn_count} legal turns in {NETWORK}")
