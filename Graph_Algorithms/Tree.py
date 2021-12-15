import networkx as nx


class TreeNode:
    def __init__(self, key: str, parent=None):
        self.__key = key
        self.__parent: TreeNode = parent
        self.__children: set[TreeNode] = set()

    def add_children(self, *nodes):
        for node in nodes:
            if isinstance(node, TreeNode):
                self.__children.add(node)
            else:
                raise ValueError('Not a Tree Node')

    @property
    def key(self):
        return self.__key

    @property
    def parent(self):
        return self.__parent

    @property
    def children(self):
        return self.__children

    def __str__(self):
        if not len(self.__children):
            return '[' + self.__key + ']'
        return '[' + self.__key + ' ' + ' '.join([str(child) for child in self.__children]) + ']'


def root_tree(graph: nx.Graph, root_key):
    root = TreeNode(root_key)
    return build_tree(graph, root)


def build_tree(graph: nx.Graph, node: TreeNode):
    for neighbor in nx.neighbors(graph, node.key):
        if node.parent and neighbor[0] == node.parent.key:
            continue
        child = TreeNode(neighbor, node)
        node.add_children(build_tree(graph, child))
    return node
