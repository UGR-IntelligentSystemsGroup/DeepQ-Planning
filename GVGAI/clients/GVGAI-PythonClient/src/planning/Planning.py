import yaml
import subprocess

class Planning:
    def __init__(self, config_path):
        # Load game information
        with open(config_path, "r") as stream:
            try:
                self.game_information = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # Command to be run
        self.planner_command = ['planning/ff', '-o', self.game_information["domainFile"],
                '-f', self.game_information["problemFile"], '-O', '-g', '1', '-h', '1'
        ]

    def generate_problem_file(self, pddl_predicates, pddl_objects, goal):
        with open(self.game_information["problemFile"], "w") as problem_file:
            # Write initial lines
            problem_file.write(f'(define (problem {self.game_information["domainName"]}Problem)\n')
            problem_file.write(f'\t(:domain {self.game_information["domainName"]})\n'.expandtabs(4))

            # Write objects
            problem_file.write("\t(:objects\n".expandtabs(4))

            for pddl_type, objects in pddl_objects.items():
                if len(objects) > 0:
                    problem_file.write(f"\t{' '.join(sorted(objects))} - {pddl_type}\n".expandtabs(8))

            problem_file.write("\t)\n".expandtabs(4))

            # Write init state
            problem_file.write("\t(:init\n".expandtabs(4))
            pattern = "\n\t"
            problem_file.write(f"\t{pattern.join(pddl_predicates)}\n".expandtabs(8))
            problem_file.write("\t)\n".expandtabs(4))

            # Write goal
            problem_file.write("\t(:goal\n".expandtabs(4))
            problem_file.write("\t(AND\n".expandtabs(8))
            problem_file.write(f"\t{goal}\n".expandtabs(12))
            problem_file.write("\t)\n".expandtabs(8))
            problem_file.write("\t)\n".expandtabs(4))


            problem_file.write(")\n")

    def call_planner(self):
        process = subprocess.Popen(self.planner_command, stdout=subprocess.PIPE)
        out = process.communicate()

        planner_output = out[0].decode('utf-8')

        return planner_output
        

