standing = [
    "chair",
    "extended hand-to-big-toe",
    "extended side angle",
    "godess",
    "half moon",
    "high lunge",
    "lord of the dance",
    "pyramid",
    "tree",
    "warrior",
    "balance",
    "camel",
    "garland",
    "standing forward bend",
]
floor = [
    "boat",
    "bow",
    "bridge",
    "crow",
    "lotus",
    "cat",
    "cobra",
    "cow",
    "downdog",
    "embryo in womb",
    "happy baby",
    "legs-up-the-wall",
    "one-legged king pigeon",
    "pigeon",
    "plank",
    "scorpion",
    "seated forward bend",
    "splits",
    "staff",
    "butterfly",
]

full_stand=[
    "chair",
    "extended hand-to-big-toe",
    "extended side angle",
    "half moon",
    "lord of the dance",
    "pyramid",
    "standing forward bend",
    "tree",
]
bend_knees = [
    "balance",
    "garland",
    "godess",
    "high lunge",
    "warrior",
]
spread_feet =[
    "one-legged king pigeon",
    "splits",
]
sitting =[
    "boat",
    "butterfly",
    "embryo in womb",
    "happy baby",
    "legs-up-the-wall",
    "lotus",
    "seated forward bend",
    "staff",
]
not_sitting=[
    "bow",
    "bridge",
    "crow",
    "cat",
    "cobra",
    "cow",
    "downdog",
    "pigeon",
    "plank",
    "scorpion",
]


levels = {
    "level1": [(standing, floor, "handstand")],
    "level2": [(full_stand, bend_knees,"camel"),(spread_feet, sitting,not_sitting)],
    "level3": [(standing, floor)],
    "level4": [(standing, floor)],
    "level5": [(standing, floor)],
}
