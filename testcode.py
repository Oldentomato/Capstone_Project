import tensorflow as tf
from tensorflow.keras.layers import Dense

# 음식 카테고리 및 구조 정의
# food_structure = {
#     "음식": {
#         "밥": ["쌀밥", "순대국밥", "카레라이스", "일반김밥"],
#         "국요리": ["라면", "떡국", "미역국", "삼계탕", "된장찌개", "돼지고기김치찌개"],
#         "반찬": ["조기찜", "돼지고기수육", "떡갈비", "소불고기", "훈제오리", "달걀말이", "소세지", "떡볶이", "소고기장조림", "닭튀김", "돈가스", "탕수육"]
#     }
# }
food_structure = {
    0: [0, 1, 2, 3],
    1: [0, 1, 2, 3, 4, 5],
    2: [0,1,2,3,4,5,6,7,8,9,10,11]
    
}

class TreeNode:
    def __init__(self, id, is_leaf=False):
        self.id = id
        self.is_leaf = is_leaf
        self.left = None
        self.right = None

    def set_children(self, left, right):
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.is_leaf

def build_food_tree_structure(node, food_structure):
    if not isinstance(food_structure, dict):
        return TreeNode(food_structure, is_leaf=True)
    
    children = []
    for category, child_structure in food_structure.items():
        child_node = build_food_tree_structure(TreeNode(category), child_structure)
        children.append(child_node)
    
    if len(children) == 1:
        return children[0]
    else:
        merged_node = TreeNode(None)
        merged_node.set_children(children[0], children[1])
        for child in children[2:]:
            merged_node = TreeNode(None).set_children(merged_node, child)
        return merged_node

root_node = build_food_tree_structure(None, food_structure)
print(root_node)

# # Define the input layer
# inputs = tf.keras.layers.Input(shape=(input_shape,))

# # Define the hidden layers
# hidden = tf.keras.layers.Dense(hidden_units, activation='relu')(inputs)

# # Define the output layer
# def traverse_tree(node, code):
#     if node is None:
#         return code
#     left_code = traverse_tree(node.left, code + [0])
#     right_code = traverse_tree(node.right, code + [1])
#     if node.is_leaf():
#         return [(node.id, code)]
#     else:
#         return left_code + right_code

# def hierarchical_softmax(inputs, tree, leaf_ids):
#     codes = traverse_tree(tree, [])
#     codes = sorted(codes, key=lambda x: x[0])
#     codes = [x[1] for x in codes]
#     weights = tf.keras.layers.Dense(len(codes), activation='softmax')(hidden)
#     output = tf.reduce_prod(tf.stack([weights[:, code] for code in codes], axis=1), axis=1)
#     return output

# output = hierarchical_softmax(hidden, tree, leaf_ids)

# # Define the model
# model = tf.keras.Model(inputs=inputs, outputs=output)

# # 데이터 준비
# # ...



# # Define the loss function and optimizer
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam()

# # Compile the model
# model.compile(loss=loss_fn, optimizer=optimizer)

# model.summary()
# # Train the model
# # model.fit(x_train, y_train, epochs=10, batch_size=batch_size)
