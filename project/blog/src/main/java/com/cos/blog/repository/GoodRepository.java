package com.cos.blog.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import com.cos.blog.model.Board;
import com.cos.blog.model.Good;
import com.cos.blog.model.Gpt;
import com.cos.blog.model.User;

public interface GoodRepository extends JpaRepository<Good, Integer>{
	Good findByUserAndBoard(User user, Board board);
}