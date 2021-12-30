#![allow(unused)]
/// Simple binary tree
use std::collections::VecDeque;

type Link<T> = Option<Box<Node<T>>>;

struct Node<T> {
    item: T,
    left: Link<T>,
    right: Link<T>,
}

pub struct Tree<T> {
    root: Link<T>,
}

struct NodeIterMut<'a, T: 'a> {
    item: Option<&'a mut T>,
    left: Option<&'a mut Node<T>>,
    right: Option<&'a mut Node<T>>,
}

enum State<'a, T: 'a> {
    Item(&'a mut T),
    Node(&'a mut Node<T>),
}

pub struct IterMut<'a, T: 'a>(VecDeque<NodeIterMut<'a, T>>);

impl<T> Tree<T> {
    pub fn iter_mut(&mut self) -> IterMut<T> {
        let mut deque = VecDeque::new();
        self.root
            .as_mut()
            .map(|root| deque.push_front(root.iter_mut()));
        IterMut(deque)
    }
}

impl<T> Node<T> {
    pub fn iter_mut(&mut self) -> NodeIterMut<T> {
        NodeIterMut {
            item: Some(&mut self.item),
            left: self.left.as_mut().map(|node| &mut **node),
            right: self.right.as_mut().map(|node| &mut **node),
        }
    }
}

impl<'a, T> Iterator for NodeIterMut<'a, T> {
    type Item = State<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        // ordinary direction of traversal (across node) is from left to right
        match self.left.take() {
            Some(node) => Some(State::Node(node)),
            None => match self.item.take() {
                Some(item) => Some(State::Item(item)),
                None => match self.right.take() {
                    Some(node) => Some(State::Node(node)),
                    None => None,
                },
            },
        }
    }
}

impl<'a, T> DoubleEndedIterator for NodeIterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        // reverse traversal (across node) is from right to left!
        match self.right.take() {
            Some(node) => Some(State::Node(node)),
            None => match self.item.take() {
                Some(item) => Some(State::Item(item)),
                None => match self.left.take() {
                    Some(node) => Some(State::Node(node)),
                    None => None,
                },
            },
        }
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.0.front_mut().and_then(|node_it| node_it.next()) {
                Some(State::Item(item)) => return Some(item),
                Some(State::Node(node)) => self.0.push_front(node.iter_mut()),
                None => {
                    if let None = self.0.pop_front() {
                        return None;
                    }
                }
            }
        }
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            match self.0.back_mut().and_then(|node_it| node_it.next_back()) {
                Some(State::Item(item)) => return Some(item),
                Some(State::Node(node)) => self.0.push_back(node.iter_mut()),
                None => {
                    if let None = self.0.pop_back() {
                        return None;
                    }
                }
            }
        }
    }
}
