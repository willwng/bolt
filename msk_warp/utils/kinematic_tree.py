from .converted_objects import JointData, BodyData
from msk_warp.utils.converted_objects import GROUND_BODY


class KinematicTreeNode:
    def __init__(self, body: BodyData, joint: JointData):
        self.body = body
        self.name = body.name
        self.joint = joint
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"Node(name={self.name}, joint={self.joint.name if self.joint else None})"

    def verify(self) -> bool:
        """ returns true if a joint exists for all non-root nodes """
        if self.joint is None and self.body != GROUND_BODY:
            raise ValueError(f"Node '{self.name}' does not have an associated joint.")
        for child in self.children:
            child.verify()
        return True

    def get_children_no_roots(self) -> list["KinematicTreeNode"]:
        """ Returns a list of direct children, unless this node is ground in which case it returns an empty list """
        if self.body == GROUND_BODY:
            return []
        return self.children


class KinematicTree:
    def __init__(self, root_body: BodyData, root_joint: JointData):
        self.root = KinematicTreeNode(root_body, root_joint)

    def _find_node(self, current_node: KinematicTreeNode, target: BodyData):
        """Recursively search for a node for a given body starting from the current node."""
        if current_node.body == target:
            return current_node
        for child in current_node.children:
            result = self._find_node(child, target)
            if result is not None:
                return result
        return None

    def add_edge(self, parent_body: BodyData, child_body: BodyData, joint: JointData):
        """ Add an edge from parent to child, storing joint as the data in child node """
        parent_node = self._find_node(self.root, parent_body)
        if parent_node is None:
            raise ValueError(f"Parent node '{parent_body.name}' not found.")

        child_node = KinematicTreeNode(body=child_body, joint=joint)
        parent_node.add_child(child_node)
        return

    def _dfs(self, node, callback: callable = None):
        """ dfs on tree, running callback on each node if provided """
        if callback is not None:
            callback(node)

        for child in node.children:
            self._dfs(child, callback=callback)
        return

    def verify(self):
        """ Verify that the tree is a valid kinematic tree (no cycles) """
        visited = set()

        def check_visited(node):
            if node.name in visited:
                raise ValueError(f"Cycle detected at node '{node.name}'")
            visited.add(node.name)

        self._dfs(self.root, callback=check_visited)
        self.root.verify()
        return

    def create_body_tree(self, names_only: bool = True) -> list[list]:
        """ Returns a list such that for each index i of the list, contains the nodes at that tree level/depth"""
        levels = []

        def add_to_levels(node, depth=0):
            if depth == len(levels):
                levels.append([])
            levels[depth].append(node.name if names_only else node)

            for child in node.children:
                add_to_levels(child, depth + 1)

        add_to_levels(self.root)
        return levels

    def forward_ordering(self) -> list[KinematicTreeNode]:
        """ Return a list of nodes in forward kinematic order, ignoring ground """
        # For a simple DFS ordering:
        # ordering = []
        # self._dfs(self.root, callback=lambda node: ordering.append(node))

        # However, it's probably more memory efficient to pack bodies in the same level together
        body_tree_levels = self.create_body_tree(names_only=False)
        ordering = [node for level in body_tree_levels for node in level]
        return ordering


    def render(self):
        """ Optionally render a graphviz of the tree """
        import graphviz
        graph = graphviz.Digraph(format='png')

        forward_ordering = self.forward_ordering()

        def get_index(node):
            return forward_ordering.index(node)

        def add_edges(node):
            for child in node.children:
                head_name = f"{node.name}\n({get_index(node)})"
                tail_name = f"{child.name}\n({get_index(child)})"
                edge_label = f"{child.joint.name}\n({child.joint.mob_type.name.lower()})"
                graph.edge(head_name, tail_name, label=edge_label)
                add_edges(child)

        add_edges(self.root)
        graph.render('kinematic_tree', view=True)
        return
