import os


class SimulationStatus:
    def __init__(
        self,
        N_turns_per_run=None,
        N_turns_target=None,
        check_for_resubmit=False,
    ):
        self.N_turns_target = N_turns_target
        self.N_turns_per_run = N_turns_per_run
        self.check_for_resubmit = check_for_resubmit
        self.filename = "simulation_status.sta"

    def dump_to_file(self):
        lines = []
        lines.append(f"present_simulation_part = {self.present_simulation_part}\n")
        lines.append(f"first_turn_part = {self.first_turn_part}\n")
        lines.append(f"last_turn_part = {self.last_turn_part}\n")
        lines.append(f"present_part_done = {self.present_part_done!r}\n")
        lines.append(f"present_part_running = {self.present_part_running!r}\n")

        with open(self.filename, "w") as fid:
            fid.writelines(lines)

    def load_from_file(self):
        ddd = {}
        with open(self.filename) as fid:
            exec(fid.read(), ddd)

        self.present_simulation_part = ddd["present_simulation_part"]
        self.first_turn_part = ddd["first_turn_part"]
        self.last_turn_part = ddd["last_turn_part"]
        self.present_part_done = ddd["present_part_done"]
        self.present_part_running = ddd["present_part_running"]

    def print_from_file(self):
        with open(self.filename) as fid:
            print(fid.read())

    def before_simulation(self):
        self.first_run = False
        try:
            self.load_from_file()
            self.present_simulation_part += 1
            self.first_turn_part += self.N_turns_per_run
            self.last_turn_part += self.N_turns_per_run
        except IOError:
            print("Simulation Status not found --> initializing simulation")
            self.first_run = True
            self.present_simulation_part = 0
            self.first_turn_part = 0
            self.last_turn_part = self.N_turns_per_run - 1
            self.present_part_done = True
            self.present_part_running = False

        if not self.present_part_done or self.present_part_running:
            raise ValueError("The previous simulation part seems not finished!!!!")

        self.present_part_done = False
        self.present_part_running = True
        self.dump_to_file()

        print("Starting part:\n\n")
        self.print_from_file()
        print("\n\n")

    def after_simulation(self):
        self.load_from_file()
        self.present_part_done = True
        self.present_part_running = False
        self.dump_to_file()

        print("Done part:\n\n")
        self.print_from_file()
        print("\n\n")

        if self.check_for_resubmit and self.last_turn_part + 1 < self.N_turns_target:
            print("resubmit the job")
            os.system("sbatch job.cmd")

    def restart_last(self):
        self.load_from_file()
        self.N_turns_per_run = self.last_turn_part - self.first_turn_part + 1

        self.present_simulation_part -= 1
        self.first_turn_part -= self.N_turns_per_run
        self.last_turn_part -= self.N_turns_per_run
        self.present_part_done = True
        self.present_part_running = False

        self.dump_to_file()

        print("Restored status:\n\n")
        self.print_from_file()
        print("\n\n")
