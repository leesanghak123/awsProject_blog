package com.cos.blog.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import com.cos.blog.model.Board;

public interface BoardRepository extends JpaRepository<Board, Integer>{
	// 아무것도 없지만 JpaRepository가 다 정보를 들고있다
	@Modifying
    @Query("update Board b set b.count = b.count + 1 where b.id = :id")
    void incrementCount(@Param("id") int id);
}
