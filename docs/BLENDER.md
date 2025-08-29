# Blender + Phobos Add-On

## Pre-requisite

- Install Blender 4.2.11 LTS
- Download the [Phobos Add-on](https://github.com/dfki-ric/phobos/releases) and import it into Blender

## Visuals

1. Look for `.stl` files online, so that you don't have to deal with creating the visuals at the very least
1. Load `.stl` files inside and make sure the import scale is set to `0.005`, this is usually a good size (keeps it under 1x1m box)
1. Set the *Phobostype* to **visual** and then set *Geometry* to **mesh**

## Collisions

### Automated Method

It is important to understand that in order to create stable collision meshes that will work in PyBullet, each collision mesh cannot be **non-convex**.

When encountering non-convex shapes, it is advised to decompose them into smaller, convex pieces. This repo provides with the `auto_collision_mesh_generator.py` script which can be used in the following format

```bash
python auto_collision_mesh_generator.py -i {input-file}.stl -o {output-file}.obj
```

However, there will be times when the script doesn't work that well (e.g. too many meshes, the tolerances are too tight, shapes are too hard, etc.), and you will have to manually resort to creating them by hand in Blender

### Manual Method

The fastest way to do this is to select the object (while in "Object Mode") that you want to create the collision mesh for, go 
into "Edit Mode", and then selecting the vertices/edges/faces (whichever you prefer) to create the separate collision meshes.
You do not need the insides of the object to have collision meshes, the surfaces of the object are enough.

### Phobostype + Geometry

You will also need to convert the *Phobostype* and the *Geometry* to **collision** and **mesh** respectively.

## Creating Link(s)

Link(s) are at the core of the `.urdf` format. Each link will contain both the visual and collision mesh that allows PyBullet to show and calculate collisions.

**Creating a Link & "Parenting" it to both visual and collision objects**:

1. Create Link(s) -> Do this from Phobos
1. Once you've created a link, it is recommended to place the link at the "3D Cursor", always place this 3D Cursor at the origin `(0,0,0)` so that the positions of the visual and collision meshes are exactly where you placed it in Blender
1. Select the visual and the collision objects that you want to parent to the link, **selecting the link last**.
1. Right click within the viewport and select "Parent" > "Object (Keep Transform)"
1. Finally, remember to set the name for the link by using "Name Model". The convention to use is `obj_i` with `i` starting from 1

## Object Placement

Place the object between `(-1,-1,0)` and `(0,0,0)`, as this the default camera view for PyBullet upon initial load.

## Exporting

**Export Settings**:

1. Select "Models" to `urdf`
1. Select "Meshes" to `obj`
1. Under "URDF Export", set "URDF ..." to "obj"

Then click, "Export Model" and tick the select all to export all link(s) that has been created.
